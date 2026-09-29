"""Data for the essentiality classifier: download, labels and cross-validation split.

Stages (see ``run_all.py``):

* ``download``: proteomes for every OGEE genome (UniProt reference proteome, else
  UniProtKB, else NCBI/custom sources), merged into ``all_sequences_folder``.
  Needs network access, ``gdown`` (OGEE files are mirrored on Google Drive) and the
  NCBI ``datasets`` CLI (``$NCBI_DATASETS``, default ``~/datasets`` then ``datasets`` on PATH).
* ``labels``: map OGEE essentiality calls onto each proteome
  (``label_folder/labeled_essentiality_taxid{t}.pkl``) and write the
  duplicate-free FASTAs (``fasta_folder``).
* ``split``: cluster all sequences with mmseqs2 at 40% identity and assign whole
  clusters to 5 folds (0 = test, 1 = validation, 2-4 = train).

Fold labels come from the TSV member ids (``cluster_fold_rows`` + ``label_folds``), so
every mmseqs cluster lands in exactly one fold. The seeded split is written to
``..._seed{S}.pkl``.
"""
import argparse
import concurrent.futures
import gzip
import json
import os
import pickle
import random
import re
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import tqdm
from Bio import SeqIO
from requests.adapters import HTTPAdapter, Retry
from requests.structures import CaseInsensitiveDict

from experiments.essentiality.common import load_config, splits_filename

re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def _import_gdown():
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is needed to download the OGEE tables, the Fitness Browser table for "
            "taxid 294 and the JCVI-Syn1.0 labels (all mirrored on Google Drive): "
            "pip install gdown") from e
    return gdown


def _ncbi_datasets() -> str:
    """Path of the NCBI `datasets` CLI: $NCBI_DATASETS, else ~/datasets, else `datasets` on PATH."""
    candidates = [os.environ.get("NCBI_DATASETS"), os.path.expanduser("~/datasets"), shutil.which("datasets")]
    for c in candidates:
        if c and os.path.exists(os.path.expanduser(c)):
            return os.path.expanduser(c)
    raise FileNotFoundError(
        "NCBI `datasets` CLI not found (set $NCBI_DATASETS; see "
        "https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/)")


# --------------------------
# Download (formerly download_ess_data.py)
# --------------------------

def download_gz_file(url: str, 
                     folder: str | os.PathLike, 
                     output_filename: Optional[str] = None) -> str:
    r"""
    Download gz file at given url, then extract into folder and delete gz file. 
    Args:
        url: url of the gz file, it should end with the filename.extension.gz
        folder: where to download and extract the .gz file
        output_filename: if given, the name of the extracted file
    
    Returns:
        output_filename: name of the extracted file
    """
    
    # Get file names from url assuming the last thing 
    # after the last "/" is the filename which ends with 
    # a ".extension.gz"
    gz_filename = os.path.join(folder, url.split("/")[-1])
    if output_filename is None:
        output_filename = ".".join(gz_filename.split(".")[:-1])

    # Download the file
    response = requests.get(url, stream=True)
    with open(gz_filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    print(f"Downloaded {gz_filename}")

    # Extract the .gz file
    with gzip.open(gz_filename, "rb") as gz_file:
        with open(output_filename, "wb") as out_file:
            shutil.copyfileobj(gz_file, out_file)

    print(f"Extracted to {output_filename}")

    # Remove the compressed file
    os.remove(gz_filename)
    print(f"Deleted {gz_filename}")

    return output_filename

def find_problematic_genomes(fasta_directory: str | os.PathLike, 
                             min_num_seq: int = 10,
                             must_include_list: Optional[List[int]] = None) -> List[int]:
    r"""
    Look at the number of sequences present in each fasta file in a directory.
    Assuming that the fasta filenames end with f"taxid{taxid}.fasta", a list of 
    the taxids that have fasta files with less than min_num_seq sequences is returned. 

    Args:
        fasta_directory: directory containing all the fasta files
        min_num_seq: minimum number of sequences for a fasta file to be OK
        must_include_list: list of taxids to include in the final list, even if 
                           the fasta file contains enough entries

    Returns:
        problematic_taxids
    """
    
    ## Find fasta files with only a few sequences
    problematic_taxids = []
    for filename in os.listdir(fasta_directory):
        if ".fasta" in filename:
            list_of_records = list(SeqIO.parse(os.path.join(fasta_directory, filename), "fasta"))
            if len(list_of_records) < min_num_seq:
                corresp_taxid = re.search(r'taxid(.*?).fasta', filename).group(1)
                print(f"Warning: the fasta file corresponding to taxid {corresp_taxid} contains {len(list_of_records)} records")
                problematic_taxids.append(int(corresp_taxid))
    
    if must_include_list is not None:
        problematic_taxids = list(set(problematic_taxids + list(must_include_list)))
    
    return problematic_taxids

def find_problematic_labels(labels_directory: str | os.PathLike, 
                            min_num_seq: int = 10) -> List[int]:
    r"""
    Look at the number of gene_names with a corresponding essentiality label
    in each label file in a directory. Assuming that the label filenames 
    end with f"taxid{taxid}.pkl", a list of the taxids that have less than 
    min_num_seq genes labelled is returned. 

    For details about the label files look at link_data_sources.process_one_fasta

    Args:
        labels_directory: directory containing all the label files
        min_num_seq: minimum number of sequences for a fasta file to be OK
        must_include_list: list of taxids to include in the final list, even if 
                           the fasta file contains enough entries

    Returns:
        unlabelled_taxids
    """
    ## Find taxids with few labels
    unlabelled_taxids = []
    for label_file in os.listdir(labels_directory):
        num_of_labels = 0
        corresp_taxid = int(re.search(r'taxid(.*?).pkl', label_file).group(1))
        with open(os.path.join(labels_directory, label_file), "rb") as f:
            label = pickle.load(f)
        
        for v in label.values():
            if v["Essentiality"]:
                num_of_labels += 1

        if num_of_labels < min_num_seq:
            unlabelled_taxids.append(corresp_taxid)
            print(f"Warning: the fasta file corresponding to taxid {corresp_taxid} contains {len(label)} records but only {num_of_labels} are labelled")

    return unlabelled_taxids

def move_all_fastas_to_one_folder(all_sequences_folder: str | os.PathLike,
                                  taxid_with_reference: Dict[str, List[int]],
                                  folders_with_reference: Dict[str, str]) -> None:
    """
    Move all data from the folders specific for their source to a common folder, 
    according to dictionary taxid_with_reference
    Args:
        all_sequences_folder: final folder to contain all sequences
        taxid_with_reference: a dictionary linking a source to a list of taxids that should come from that source
        folders_with_reference: a dictionary linking a source to the path of the folder containing data from that source
    """

    print(f"Moving all data to {all_sequences_folder}")

    if not os.path.exists(all_sequences_folder):
        os.mkdir(all_sequences_folder)

    for source, taxid_from_source in taxid_with_reference.items():
        for taxid in tqdm.tqdm(taxid_from_source, desc=f"{source} seqs", ncols=100):
            
            filename = f"{source}_data_taxid{taxid}.fasta" if source != "uniprot-reference-genomes" else f"uniprotkb_data_taxid{taxid}.fasta"
            
            source_path = os.path.join(folders_with_reference[source], filename)
            destination_path = os.path.join(all_sequences_folder, filename)
            
            if not os.path.exists(destination_path):
                shutil.copy2(source_path, destination_path)

def get_next_link(headers: CaseInsensitiveDict) -> str:
    r"""
    Starting from the header of link of "search" type, returns the next link
    to be visited (i.e. the link of the next batch).

    code adapted from Uniprot help page (https://www.uniprot.org/help/api_queries)
    """
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url: str) -> Iterator[requests.models.Response]:
    r"""
    Generic function that returns a batch (a page in the paginated search 
    results from uniprot) starting from a link of "search" type.

    code adapted from Uniprot help page (https://www.uniprot.org/help/api_queries)
    """
    assert "search" in batch_url, "The url should be of search type"

    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        yield response
        batch_url = get_next_link(response.headers)

def download_data_from_uniprot(directory: str | os.PathLike, 
                               taxid_set: List[int]) -> None:
    r"""
    Download proteomes corresponding to given list of taxonomy IDs from Uniprot as 
    fasta files (if the path f"{directory}/uniprotkb_data_taxid{tax_id}.fasta" doesn't 
    already exist). For each taxid, first look for a proteome matching the ID. If the 
    proteome is not found, look for all sequences in UniprotKB matching the ID. 

    Args:
        directory: where to save all the fasta files as f"uniprotkb_data_taxid{tax_id}.fasta"
        taxid_set: the list of NCBI taxonomy IDs to download the proteomes of 
    """

    if not os.path.exists(directory):
        os.mkdir(directory)

    genes_per_batch = 500
    counter = 0

    for tax_id in tqdm.tqdm(taxid_set, leave=False, ncols=100, desc="Uniprot"):
        
        taxid_fastafilename = os.path.join(directory, f"uniprotkb_data_taxid{tax_id}.fasta")
        
        if not os.path.exists(taxid_fastafilename):

            proteome_info = find_uniprot_proteome_id(tax_id, proteome_type = None)
            if proteome_info is not None:
                _, proteome_id = proteome_info
                url = f"https://rest.uniprot.org/uniprotkb/search?compressed=false&format=fasta&query=%28proteome%3A{proteome_id}%29&size={genes_per_batch}"
            else:
                url = f"https://rest.uniprot.org/uniprotkb/search?compressed=false&format=fasta&query=%28%28taxonomy_id%3A{tax_id}%29%29&size={genes_per_batch}"
            
            with open(taxid_fastafilename, 'w') as f:
                for batch in get_batch(url):
                    for line in batch.text.splitlines():
                        print(line, file=f)

            counter += 1

    print(f"Downloaded {counter} fasta files from uniprot.\n{len(taxid_set)-counter} files already present at {directory}")

def find_uniprot_proteome_id(taxonomy_id: int, proteome_type: Optional[int] = None) -> Tuple[str, str] | None:
    """
    Find superkingdom and uniprot proteome id of a NCBI taxonomy ID. 
    Args:
        taxonomy_id
        proteome_type: if == 1 then look only for reference proteome
    Returns:
        superkingdom
        proteome_id: Uniprot proteome ID
    """

    base_url = "https://rest.uniprot.org/proteomes/search"

    query = "".join([f"taxonomy_id:{taxonomy_id}", 
                f" AND proteome_type:1" if proteome_type==1 else ""])
    params = {
        "query": query,
        "format": "json",
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    results = data.get('results', [])
    if results != []:
        superkingdom = [entry['superkingdom'] for entry in data.get('results', [])].pop()
        proteome_ids = [entry['id'] for entry in data.get('results', [])]
        proteome_id = proteome_ids.pop()

        return superkingdom, proteome_id
    else:
        print(f"No proteome found corresponding to taxid {taxonomy_id}")
        return None

def get_reference_proteome_gz_link(taxonomy_id: str) -> str | None:
    """
    Find link to gz file containing uniprot reference proteome associated to NCBI taxonomy ID.
    Args:
        taxonomy_id
    Returns:
        gz_link if reference proteome is in uniprot else None
    """
    proteome_info = find_uniprot_proteome_id(taxonomy_id, proteome_type = 1)
    if proteome_info is not None:
        superkingdom, proteome_id = proteome_info
    else:
        return None
    return f"https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/{superkingdom.capitalize()}/{proteome_id}/{proteome_id}_{taxonomy_id}.fasta.gz"

def get_uniprot_reference_proteome(directory: str | os.PathLike, 
                                   tax_id: int, 
                                   overwrite: bool = False) -> None | int:
    """
    Download uniprot reference proteome into f"{directory}/uniprotkb_data_taxid{tax_id}.fasta".
    Args:
        directory: where to save the fasta file
        tax_id: NCBI taxonomy ID
        overwrite: whether to overwrite the file if already present
    Returns:
        None if successful, otherwise returns tax_id
    """
    
    output_file = os.path.join(directory, f"uniprotkb_data_taxid{tax_id}.fasta")

    if os.path.exists(output_file) and (overwrite == False):
        print(f"File already present at {output_file}")
        return
    
    url = get_reference_proteome_gz_link(tax_id)
    if url is None:
        return tax_id
    
    try:
        filename = download_gz_file(url=url, folder=directory)
        subprocess.run(f"mv {filename} {directory}/uniprotkb_data_taxid{tax_id}.fasta", shell=True, check=True)
    except:
        return tax_id

def get_dataset_info_df(ogee_data_directory: str | os.PathLike) -> pd.DataFrame:
    """
    Download essentiality data from OGEE and load the datasets data into a dataframe

    Args:
        ogee_data_directory: where to save the data
    """
    
    # Download the OGEE data previously saved on google drive (because the OGEE website has an expired certificate)
    os.makedirs(ogee_data_directory, exist_ok=True)

    ogee_filenames = {"gene_essentiality.txt": "19a_33lnOZ0X8DCB6DL-bKNUtMnmRVO7o",
    "genes.txt": "1LrnT6EOiWg6wk5IzMXdlYYBc44Oig4V1",
    "datasets.txt": "1h-QGDQAI_u0hWg_epl7XmYjLw5bPes8k"}

    for file, google_drive_id in ogee_filenames.items():
        file_fullpath = os.path.join(ogee_data_directory, file)
        if not os.path.exists(file_fullpath):
            _import_gdown().download(id=google_drive_id, output=file_fullpath)
            
    # Import the datasets info data using pandas
    dataset_info_df = pd.read_csv(ogee_data_directory+"/datasets.txt", sep="\t", encoding_errors="replace", usecols=["datasetID", "taxID", "url"])
    
    return dataset_info_df

def get_data_from_othersources(taxid: int, 
                               out_directory: str | os.PathLike) -> None:
    """
        Manual download methods for some taxids.
        Args:
            taxid
            out_directory
    """
    
    if taxid == 580240:
        if not os.path.exists(os.path.join(out_directory, f"othersource_data_taxid{taxid}.fasta")):
            url = "http://sgd-archive.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans.fasta.gz"
            filename = download_gz_file(url=url, folder=out_directory)
            subprocess.run(f"mv {filename} {out_directory}/othersource_data_taxid{taxid}.fasta", shell=True, check=True)
        else:
            print(f"File already present at {os.path.join(out_directory, f'othersource_data_taxid{taxid}.fasta')}")
    elif taxid == 294:
        file, google_drive_id = ("taxid294_data.xlsx", "1opb_eGqJfMZY__F-swCH7g4az-aGSKx9")
        file_fullpath = os.path.join(out_directory, file)
        if not os.path.exists(file_fullpath):
            _import_gdown().download(id=google_drive_id, output=file_fullpath)

        df = pd.read_excel(file_fullpath, sheet_name=0, skiprows=13)
        relevant_df = df.loc[df["organism"].str.contains(f"Pseudomonas fluorescens FW300-N1B4"), ["locusId", "protein_id"]]

        download_data_from_ncbi(
            taxid=294,
            folder_path=out_directory,
            output_filename="othersource_data_taxid294.fasta",
            accession="GCF_001625455.1",
            overwrite=True,
            get_gene_name_from_cds=False)

        fasta_fullpath = os.path.join(out_directory, "othersource_data_taxid294.fasta")
        all_records = []
        for record in SeqIO.parse(fasta_fullpath, format="fasta"):
            matching_id = relevant_df.loc[relevant_df["protein_id"]==record.id, "locusId"].tolist()
            assert len(matching_id) <= 1, f"(Taxid {taxid}) Found more than one locus {matching_id} corresponding to protein id {record.id}, maybe coming from different strains?"
            if len(matching_id) == 1:
                record.id = matching_id[0]
            all_records.append(record)

        SeqIO.write(sequences=all_records, handle=fasta_fullpath, format="fasta")
        subprocess.run(f"rm {out_directory}/taxid294_data.xlsx", shell=True, check=True)
    elif taxid in [290336, 709015, 93064, 1380365, 1122134, 321846]:
        org_id_fitgen = {
            679895:"Keio",
            290336:"Koxy",
            709015:"Ponti", 
            93064:"Korea",  
            1380365:"Dyella79", 
            1122134:"Kang", 
            321846:"WCS417"
        }

        url = f"https://fit.genomics.lbl.gov/cgi-bin/orgSeqs.cgi?orgId={org_id_fitgen[taxid]}"

        if not os.path.exists(f"{out_directory}/othersource_data_taxid{taxid}.fasta"):
            response = session.get(url, stream=True)
            with open(f"{out_directory}/othersource_data_taxid{taxid}.fasta", "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        else:
            print(f"File already present at {os.path.join(out_directory, f'othersource_data_taxid{taxid}.fasta')}")
    elif taxid in [679895, 400667, 220341, 243273]:
        accessions = {400667: "GCA_000015425.1",
                      220341: 'GCF_000195995.1', 
                      243273: 'GCF_000027325.1', 
                      679895: 'GCF_000750555.1'}
        
        download_data_from_ncbi(
            taxid=taxid,
            folder_path=out_directory,
            output_filename=f"othersource_data_taxid{taxid}.fasta",
            accession=accessions[taxid],
            overwrite=True,
            get_gene_name_from_cds=True)
    else:
        print(f"No download method for taxid {taxid} --> Download manually!")

def get_ncbi_genome_accession(taxid: int, 
                              assembly_level: Optional[str] = None) -> str | None:
    """
    Get NCBI genome accession from NCBI taxonomy ID using the NCBI Datasets command line tool,
    installed in the home folder. 

    Args:
        taxid: the NCBI taxonomy ID
        assembly_level: "complete" or None
    Returns:
        accession: NCBI genome accession if there is at least one 'current' accession, otherwise return None
    """
    datasets_bin = _ncbi_datasets()

    if assembly_level == "complete":
        ncbi_download_command = f"{datasets_bin} summary genome taxon {taxid} --assembly-level complete --assembly-version 'latest' --as-json-lines --tax-exact-match"
    elif assembly_level == None:
        ncbi_download_command = f"{datasets_bin} summary genome taxon {taxid} --assembly-version 'latest' --as-json-lines --tax-exact-match"
    else:
        raise ValueError
    
    # Run the command and capture output
    genome_summary = subprocess.run(ncbi_download_command, shell=True, capture_output=True, text=True, check=True)
    outputs = genome_summary.stdout.strip().split('\n') # Split the returned JSON objects
    for i in range(len(outputs)):
        outputs_as_json = json.loads(outputs[i]) # Parse the first JSON object (RefSeq)
        status = outputs_as_json.get("assembly_info").get("assembly_status")
        if status != "suppressed":
            accession = outputs_as_json.get("accession")
            return accession
        
    return None

def download_data_from_ncbi(folder_path: str | os.PathLike,
                            taxid: int,
                            output_filename: str,
                            get_gene_name_from_cds: bool = False,
                            accession: Optional[str] = None,
                            overwrite: bool = False) -> None:
    """
    Download the proteome corresponding to a NCBI taxonomy ID using NCBI datasets 
    command line tool saved in home folder. 

    Args:
        folder_path: complete path of the folder where to dowload the fasta file
        taxid: NCBI taxonomy ID to download
        output_filename: name to give to the file
        get_gene_name_from_cds: if True, after having downloaded the fasta file,
                                parse it and for each gene add the CDS gene name 
                                as the protein id in the fasta header
        accession: if not None, download this accession (it overwrites the taxid)
        overwrite: if True, overwrite fasta file if already present at f"{folder_path}/{output_filename}"
    """
    datasets_bin = _ncbi_datasets()
    
    if os.path.exists(os.path.join(folder_path, output_filename)) and not overwrite:
        return
    
    if accession is None:
        # First try to get the complete assembly
        try:
            accession = get_ncbi_genome_accession(taxid, assembly_level = "complete")
        except:
            try: # Try not complete assembly
                accession = get_ncbi_genome_accession(taxid, assembly_level = None)
            except:
                print("Genome not available on NCBI")
                return

    ncbi_download_command = f"{datasets_bin} download genome accession {accession} --include protein,cds,gtf"

    if f" --filename {folder_path}/ncbi_tmp.zip" not in ncbi_download_command:
        ncbi_download_command += f" --filename {folder_path}/ncbi_tmp.zip"
    unzip_command = f"unzip {folder_path}/ncbi_tmp.zip -d {folder_path}/ncbi_tmp"

    subprocess.run(ncbi_download_command, shell=True, check=True)
    subprocess.run(unzip_command, shell=True, check=True)

    subfolders = os.listdir(os.path.join(folder_path, "ncbi_tmp/ncbi_dataset/data"))
    chosen = [folder for folder in subfolders if ("GCF_" in folder) or ("GCA_" in folder)].pop()
    print(f"\n{subfolders} --> {chosen}")

    for filename in ["protein.faa", "cds_from_genomic.fna", "genomic.gtf"]:
        destination_full_filename = f"{folder_path}/{filename}"
        subprocess.run(f"mv {folder_path}/ncbi_tmp/ncbi_dataset/data/{chosen}/{filename} {destination_full_filename}", shell=True, check=True)
    
    subprocess.run(f"rm {folder_path}/ncbi_tmp.zip", shell=True, check=True)
    subprocess.run(f"rm -r {folder_path}/ncbi_tmp", shell=True, check=True)

    if get_gene_name_from_cds:
        # Assign new gene ID based on CDS and not on the fasta file
        
        nucleotide_id_to_locus = {}
        for record in SeqIO.parse(os.path.join(folder_path, "cds_from_genomic.fna"), format="fasta"):
            locus_tags = []
            for descriptor in ["locus_tag", "gene", "gene_synonym"]:
                match = re.search(rf"\[{re.escape(descriptor)}=(.*?)\]", record.description)
                if match:
                    locus_tags.append(match.group(1))

                    if descriptor == "locus_tag":
                        
                        find_tags_command = f"""grep 'locus_tag "{match.group(1)}"; old_locus_tag' {os.path.join(folder_path, "genomic.gtf")}"""
                        try:
                            line_with_old_tags = subprocess.run(find_tags_command, shell=True, capture_output=True, text=True, check=True)
                            match = re.search(r"""old_locus_tag "(.*?)";""", line_with_old_tags.stdout)
                            if match:
                                locus_tags.extend(match.group(1).split(","))
                        except:
                            pass
                        
            nucleotide_id_to_locus[record.id] = " ".join(locus_tags)

        new_records = []
        for record in SeqIO.parse(os.path.join(folder_path, "protein.faa"), format="fasta"):
            new_id = [locus_tag for nucleotide_id, locus_tag in nucleotide_id_to_locus.items() if record.id in nucleotide_id]
            assert len(new_id) >= 1, f"No nucleotide records found with protein_id {record.id}"
            if len(new_id) == 1:
                new_id = new_id.pop()
            else:
                all_ids = []
                for id in new_id:
                    all_ids.extend(id.split(" "))
                new_id = " ".join(set(all_ids))
            
            record.id = new_id
            new_records.append(record)

        SeqIO.write(new_records, os.path.join(folder_path, output_filename), "fasta")
    else:
        subprocess.run(f"cp {folder_path}/protein.faa {folder_path}/{output_filename}", shell=True, check=True)
    
    for filename in ["protein.faa", "cds_from_genomic.fna", "genomic.gtf"]:
        subprocess.run(f"rm {folder_path}/{filename}", shell=True, check=True)

def download_ncbi_data_from_taxids(folder_path: str | os.PathLike, 
                                   list_of_taxids: List[int],
                                   get_gene_name_from_cds: bool = False,) -> None:
    """
    Download proteome of list of NCBI taxonomy IDs from NCBI. For each taxid the
    path of the fasta file will be f"{folder_path}/ncbi_data_taxid{taxid}.fasta"
    Args:
        folder_path: where to download the fasta files
        list_of_taxids: NCBI taxonomy IDs
        get_gene_name_from_cds: whether to use gene names from CDS instead of the one in the fasta
    """
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    for taxid in list_of_taxids:

        if not os.path.exists(f"{folder_path}/ncbi_data_taxid{taxid}.fasta"):
            
            download_data_from_ncbi(folder_path=folder_path,
                                    taxid=taxid,
                                    output_filename=f"ncbi_data_taxid{taxid}.fasta",
                                    get_gene_name_from_cds=get_gene_name_from_cds)
            
            print("---------------------------")
        else:
            print(f"File already present at {folder_path}/ncbi_data_taxid{taxid}.fasta")

def find_NCBI_taxids(problematic_taxids: List[int], 
                     dataset_info_df: pd.DataFrame, 
                     taxids_with_custom_download: List[int]) -> List[int]:
    """
    Starting from a list of taxids, look them up in the datasets data from OGEE, then
    return the list of taxids for which OGEE points towards NCBI.
    Args:
        problematic_taxids: the list of taxids
        dataset_info_df: the OGEE datasets data
        taxids_with_custom_download: list of taxids to exclude from this search, 
                                     because they have a custom download method
    """
    
    NCBI_taxids = []
    for taxid in problematic_taxids:
        suggested_datasource = dataset_info_df[dataset_info_df["taxID"] == taxid]["url"].values[0]
        if ("ncbi" in suggested_datasource) and (taxid not in taxids_with_custom_download):
            NCBI_taxids.append(taxid)
        #else:
        #    print(f"For {taxid} suggested {suggested_datasource}")
    print(f"For {NCBI_taxids} suggested NCBI")

    # Ensure that NCBI taxids don't overlap with custom download taxids
    for taxid in taxids_with_custom_download:
        while taxid in NCBI_taxids:
            NCBI_taxids.remove(taxid)

    return NCBI_taxids

def download_all_training_data(uniprot_reference_genomes_dir,
                               uniprot_data_dir,
                               ncbi_directory,
                               ogee_data_dir,
                               label_directory,
                               should_check_labels,
                               all_sequences_folder,
                               taxids_with_custom_download):

    folders_with_reference = {
        "uniprot-reference-genomes": uniprot_reference_genomes_dir,
        "uniprotkb": uniprot_data_dir,
        "ncbi": ncbi_directory,
        "othersource": ncbi_directory
    }

    taxid_with_reference = {
        "uniprot-reference-genomes": None,
        "uniprotkb": None,
        "ncbi": None,
        "othersource": None
    }

    dataset_info_df = get_dataset_info_df(ogee_data_dir)
    taxid_set = set(dataset_info_df["taxID"].to_list())

    remaining_taxids = []
    for taxid in taxid_set:
        taxid_or_none = get_uniprot_reference_proteome(uniprot_reference_genomes_dir, taxid, overwrite = False)
        if taxid_or_none is not None:
            remaining_taxids.append(taxid)

    print(f"Not found reference genome for taxids {remaining_taxids}")
    
    taxid_with_reference["uniprot-reference-genomes"] = list(taxid_set-set(remaining_taxids)-set(taxids_with_custom_download))
    
    # Step 3: Download the Uniprot KB fasta data corresponding to the tax IDs in OGEE
    download_data_from_uniprot(uniprot_data_dir, list(set(remaining_taxids)-set(taxids_with_custom_download)))

    print("---------------------------")

    # Step 5: check that fasta files from uniprot have enough records
    problematic_taxids = find_problematic_genomes(fasta_directory=uniprot_data_dir,
                                                must_include_list=taxids_with_custom_download)
    
    taxid_with_reference["uniprotkb"] = list(set(remaining_taxids)-set(problematic_taxids))

    print("---------------------------")

    # Step 7: find which taxID were taken from NCBI in the OGEE dataset
    NCBI_taxids = find_NCBI_taxids(problematic_taxids, dataset_info_df, taxids_with_custom_download)
    taxid_with_reference["ncbi"] = NCBI_taxids

    print("---------------------------")

    # Step 8: Download data from NCBI
    download_ncbi_data_from_taxids(ncbi_directory, NCBI_taxids, get_gene_name_from_cds=True)

    print("---------------------------")

    # Step 9: deal with the remaining problematic taxids
    very_problematic_taxids = list(set(problematic_taxids) - set(NCBI_taxids))
    if os.path.exists(label_directory) and should_check_labels:
        problematic_labels = find_problematic_labels(label_directory)
        very_problematic_taxids = list(set(very_problematic_taxids + problematic_labels))

    taxid_with_reference["othersource"] = very_problematic_taxids
    
    for taxid in very_problematic_taxids:
        get_data_from_othersources(taxid, ncbi_directory)

    print("---------------------------")

    print(taxid_with_reference)
    
    print("---------------------------")

    # Step 10: merge all datasources
    move_all_fastas_to_one_folder(all_sequences_folder = all_sequences_folder,
                                taxid_with_reference = taxid_with_reference,
                                folders_with_reference = folders_with_reference)
    
    return taxid_with_reference


# --------------------------
# Labels (formerly link_data_sources.py)
# --------------------------

def look_for_gene_oln(uniprot_entry: str, return_json = False) -> List[Dict[str, str]]:
    """
    Get gene ordered locus name (OLN) from the uniprot entry. 

    The fasta files from uniprot do not contain the gene locus but only the gene name (most of the time)
    Since in the OGEE dataset there are a lot of entries which have only the locus and not the gene
    it's necessary to get this additional data from the uniprot API.
    """

    json_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_entry}?format=json&fields=gene_primary,gene_synonym,gene_oln,gene_orf,protein_name"
    
    try:
        json_response = session.get(json_url)
        json_response.raise_for_status()
    except:
        print(f"Couldn't get json file for uniprot entry {uniprot_entry}")
        raise
    
    json_data = json_response.json() 
    
    # Get the Ordered Locus Name (OLN) from json file
    gene_oln = json_data.get("genes", [{}])[0].get("orderedLocusNames", [])

    if return_json == False:
        return gene_oln
    else:
        return gene_oln, json_data

def get_gene_names(myrecord: Dict[str, Dict[str, Any]], # myrecord is NOT a Biopython record, it's the return value of `get_records_without_duplicates`
                   source: str) -> List[str]:
    
    if source == "uniprotkb":
        all_gene_names = []
        for description in myrecord["descriptions"]:
            uniprot_entry = description.split("|")[1]

            _, json_data = look_for_gene_oln(uniprot_entry, return_json=True) # Get the Ordered Locus Name (OLN)

            # Extract Gene Names
            gene_info = json_data.get("genes", [])
            
            for gene in gene_info:
                if "geneName" in gene:
                    all_gene_names.append(gene["geneName"]["value"])  # Main gene name
                if "synonyms" in gene:
                    all_gene_names.extend([syn["value"] for syn in gene["synonyms"]])  # Alternative names
                if "orderedLocusNames" in gene:
                    all_gene_names.extend([oln["value"] for oln in gene["orderedLocusNames"]])  # Alternative names
                if "orfNames" in gene:
                    all_gene_names.extend([orf["value"] for orf in gene["orfNames"]])  # Alternative names
    else:
        all_gene_names = []
        for description in myrecord["descriptions"]:
            desc = description.split(" ")
            if source == "othersource":
                if "SGDID" in desc[2]:
                    gene_names = desc[:2] + [desc[2].split(":")[1].strip(",")] # Strip added later
                elif ":GFF" in desc[0]:
                    gene_names = [desc[0].split(":")[1]] + [desc[1]]
                elif "BW25113_" in description:
                    gene_names = []
                    for d in desc:
                        if "WP_" in d:
                            break
                        gene_names.append(d)
                else:
                    gene_names = desc[:2]
            elif source == "ncbi":
                gene_names = []
                for i in range(len(desc)):
                    gene_names.append(desc[i])
                    if "WP_" in desc[i]: # Warning! This only works for Prokaryotes
                        break
            else:
                raise Exception
            all_gene_names.extend(gene_names)

    return all_gene_names

def get_records_without_duplicates(fasta_file: str, 
                                   max_esmc_length: Optional[int] = None
                                   ) -> Tuple[Dict[str, Dict[str, Any]], List[Any]]:

    seen = {}
    records = {}
    records_to_save = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        visible_sequence = str(record.seq)[:max_esmc_length]
        
        if visible_sequence not in seen.keys():
            seen.update({visible_sequence: record.id})
            records.update({record.id: {"seq": record.seq, "descriptions": [record.description]}})
            records_to_save.append(record)
        else:
            other_id = seen[visible_sequence]
            records[other_id]["descriptions"].append(record.description)

    return records, records_to_save

def from_fasta_to_labels(taxid: int, 
                         fasta_file: str, 
                         relevant_ess_df: pd.DataFrame, 
                         source: str, 
                         return_noduplicates: bool,
                         max_esmc_length: int = None) -> Dict[str, Dict[str, Any]]:
    
    genes: Dict[str, Dict[str, Any]] = {} # The dictionary we will populate with the following loop

    dict_of_records, records_to_save = get_records_without_duplicates(fasta_file, max_esmc_length)

    for record_id, record in dict_of_records.items(): #tqdm(dict_of_records.items(), total=len(dict_of_records), ncols = 80, leave=False):
        gene_names = get_gene_names(record, source)

        maybe_alternatives = []
        for name in gene_names:
            if "_" in name:
                maybe_alternatives.append(name.replace("_",""))
        
        gene_names.extend(maybe_alternatives)

        if gene_names:
            query_for_ess_dataframe = " | ".join([f"""gene=="{name}" """ for name in gene_names])
            matching_rows = relevant_ess_df.query(query_for_ess_dataframe)
            ess = matching_rows["essentiality"].to_list()
            
            if not ess:
                attempt2_query_for_ess_dataframe = " | ".join([f"""locus=="{name}" """ for name in gene_names])
                matching_rows = relevant_ess_df.query(attempt2_query_for_ess_dataframe)
                ess = matching_rows["essentiality"].to_list()
            
            relevant_ess_df.drop(index = matching_rows.index, inplace=True)
        else:
            ess = []
    
        genes[record_id] = {
            "tax id": taxid,
            "gene": gene_names.pop() if gene_names else None,
            "synonims": gene_names if gene_names else None,
            #"UniprotEntry": uniprot_entry,
            "Essentiality": ess
        }

    if return_noduplicates:
        return genes, records_to_save
    else:
        return genes, None

def produce_labels_for_many_fastas(essentiality_df: pd.DataFrame, 
                                   all_fasta_folder: str, 
                                   output_label_folder: str,
                                   output_fasta_folder_noduplicates: Optional[str] = None,
                                   overwrite: Optional[bool] = None,
                                   max_workers: int = 4,
                                   max_esmc_length: Optional[int] = None) -> None:
    
    should_output_fasta_noduplicates = False
    if output_fasta_folder_noduplicates is not None:
        if not os.path.exists(output_fasta_folder_noduplicates):
            os.makedirs(output_fasta_folder_noduplicates)
            assert os.path.isdir(output_fasta_folder_noduplicates)
        should_output_fasta_noduplicates = True
        # else:
        #     is_the_folder_empty = (len(os.listdir(output_fasta_folder_noduplicates)) == 0)
        #     if is_the_folder_empty or (not is_the_folder_empty and overwrite):
        #         should_output_fasta_noduplicates = True
            
    files = os.listdir(all_fasta_folder)
    pattern = re.compile(r'^(.*?)_data_taxid(\d+)\.fasta$')

    #for filename in tqdm(files, ncols=100, desc="TaxIDs"):
    #    process_one_fasta(filename = filename,
    #                  pattern = pattern,
    #                  essentiality_df = essentiality_df,
    #                  all_fasta_folder = all_fasta_folder,
    #                  output_label_folder = output_label_folder,
    #                  should_output_fasta_noduplicates = should_output_fasta_noduplicates,
    #                  output_fasta_folder_noduplicates = output_fasta_folder_noduplicates,
    #                  overwrite = overwrite)

    arguments = [
        (
            filename,
            pattern,
            essentiality_df,
            all_fasta_folder,
            output_label_folder,
            should_output_fasta_noduplicates,
            output_fasta_folder_noduplicates,
            overwrite,
            max_esmc_length,
        )
        for filename in files
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_fasta, *args) for args in arguments]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

def process_one_fasta(filename: str,
                      pattern: re.Pattern,
                      essentiality_df: pd.DataFrame,
                      all_fasta_folder: str,
                      output_label_folder: str,
                      should_output_fasta_noduplicates: bool,
                      output_fasta_folder_noduplicates: Optional[str],
                      overwrite: bool,
                      max_esmc_length: int = None) -> str:
    
    match = pattern.match(filename)
    if match:
        source = match.group(1)
        taxid = int(match.group(2))
    else:
        return f"Warning: wrong filename format in {filename}"
    
    print(f"Started processing taxid {taxid}")

    output_label_filename = os.path.join(output_label_folder, f"labeled_essentiality_taxid{taxid}.pkl")
    if (not os.path.exists(output_label_filename)) or overwrite:
        # We have a fasta_file for each taxid (see file `download_ess_data.py`)
        fasta_file = os.path.join(all_fasta_folder, f"{source}_data_taxid{taxid}.fasta")

        relevant_ess_df = essentiality_df[essentiality_df["taxaID"] == taxid][["gene", "locus", "essentiality"]].copy()

        genes, records_to_save = from_fasta_to_labels(taxid, 
                                                        fasta_file, 
                                                        relevant_ess_df, 
                                                        source, 
                                                        return_noduplicates=should_output_fasta_noduplicates,
                                                        max_esmc_length=max_esmc_length)

        with open(output_label_filename, "wb") as f:
            pickle.dump(genes, f)

        if should_output_fasta_noduplicates:
            output_fasta_noduplicates_filename = os.path.join(output_fasta_folder_noduplicates, 
                                                                f"{source}_data_taxid{taxid}.fasta")
            SeqIO.write(records_to_save, output_fasta_noduplicates_filename, "fasta")

        return f"Processed: {output_label_filename}"
    else:
        return f"Skipped (exists): {output_label_filename}"


# --------------------------
# FASTA helpers (formerly dataset_utils.py)
# --------------------------

def load_fasta(fasta_file: str) -> Tuple[List[str], List[str]]:
    r"""
    Args:
        fasta_file: file containing the fasta sequences
    
    Returns:
        (seq_id, sequences), where seq_id is a list of the unique sequence
        identifiers contained in the fasta headers, and sequences is a list
        of all the sequences
    """
    seq_id = []
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id.append(record.id)
        sequences.append(record.seq)
    return seq_id, sequences

def merge_different_fasta_files(input_dir: str, output_file: str) -> None:
    r""" 
    Starting from a directory containing many fasta files to be concatenated,
    using the Bio library to ensure correct notation, concatenate all the files
    into a single one.

    Args:
        input_dir: directory containing the fasta files to be merged
        output_file: the file that will contain all the fasta entries
    """
    with open(output_file, "w") as outfile:
        for filename in tqdm.tqdm(os.listdir(input_dir), ncols=80):
            if filename.endswith(".fasta") or filename.endswith(".fa"):
                filepath = os.path.join(input_dir, filename)
                for record in SeqIO.parse(filepath, "fasta"):
                    SeqIO.write(record, outfile, "fasta")


# --------------------------
# Cross-validation split (formerly dataset_utils.split_for_crossval)
# --------------------------

def read_mmseqs_clusters(tsv_file: str) -> Tuple[List[str], List[str]]:
    """Read an mmseqs ``*_cluster.tsv`` (representative, member) as two string lists.

    Rows are grouped by representative, *not* in FASTA order.
    """
    tsv = pd.read_table(tsv_file, names=["clusters", "id"], dtype=str, keep_default_na=False)
    return tsv.clusters.tolist(), tsv.id.tolist()


def cluster_fold_rows(cluster_reps: Sequence[str], n_splits: int = 5,
                      rng: Optional[random.Random] = None) -> List[List[int]]:
    """Greedy cluster-to-fold assignment of the original code, on TSV row indices.

    Clusters are visited in a random order; fold i receives whole clusters until it
    holds at least 1/n_splits of all rows, and the last fold takes every remaining
    cluster. Returns, for each fold, the TSV row indices it contains.
    """
    rng = rng if rng is not None else random.Random()
    rows_of: Dict[str, List[int]] = {}
    for row, rep in enumerate(cluster_reps):
        rows_of.setdefault(rep, []).append(row)
    unique_clusters = list(rows_of)  # order of first appearance, as pandas .unique()
    n_rows = len(cluster_reps)
    min_ratio = 1 / n_splits
    idx = list(range(len(unique_clusters)))
    rng.shuffle(idx)
    splits: List[List[int]] = [[] for _ in range(n_splits)]
    for i in range(n_splits):
        if len(idx) == 0:
            break
        while (i == n_splits - 1) or len(splits[i]) < min_ratio * n_rows:
            if len(idx) == 0:
                break
            c = unique_clusters[idx.pop()]
            splits[i] += rows_of[c]
    return splits


def label_folds(fold_rows: Sequence[Sequence[int]], ids: Sequence[str]) -> Dict[str, int]:
    """``{ids[row]: fold}``. Pass the TSV member column: FASTA record order differs from
    the TSV's, which is grouped by cluster representative."""
    labelling = {}
    for fold, rows in enumerate(fold_rows):
        for row in rows:
            labelling[ids[row]] = fold
    return labelling


def assign_folds(cluster_reps: Sequence[str], cluster_members: Sequence[str],
                 n_splits: int = 5, seed: int = 0) -> Dict[str, int]:
    """Seeded fold assignment: every mmseqs cluster lands in exactly one fold."""
    rows = cluster_fold_rows(cluster_reps, n_splits=n_splits, rng=random.Random(seed))
    return label_folds(rows, cluster_members)


def write_indexed_fasta(input_file: str, output_file: str) -> List[str]:
    """Copy ``input_file`` with headers replaced by the record index (0, 1, ...).

    mmseqs rewrites some headers (``sp|P12345|NAME_ORG`` becomes ``P12345``), so it
    clusters this copy and its TSV ids are mapped back by index. Returns the ids.
    """
    ids = []
    with open(output_file, "w") as out:
        for i, record in enumerate(SeqIO.parse(input_file, "fasta")):
            ids.append(record.id)
            out.write(f">{i}\n{record.seq}\n")
    return ids


def mmseqs_to_fasta_ids(tsv_ids: Sequence[str], fasta_ids: Sequence[str]) -> List[str]:
    """Map the ids of an mmseqs TSV built on the raw FASTA back to FASTA record ids
    (for ``--cluster-tsv``): exact match, else the UniProt accession of ``sp|ACC|...``
    / ``tr|ACC|...`` headers, which is what mmseqs writes for them."""
    lookup: Dict[str, str] = {}
    for fid in fasta_ids:
        lookup.setdefault(fid, fid)
        m = re.match(r"^(?:sp|tr)\|([^|]+)\|", fid)
        if m:
            lookup.setdefault(m.group(1), fid)
    unknown = [t for t in tsv_ids if t not in lookup]
    if unknown:
        raise ValueError(f"{len(unknown)} mmseqs ids match no FASTA record (e.g. {unknown[:3]})")
    return [lookup[t] for t in tsv_ids]


def run_mmseqs_clustering(input_fasta: str, out_dir: str, threshold: int,
                          mmseqs: str = "mmseqs", threads: int = 24) -> str:
    """``mmseqs easy-cluster --min-seq-id threshold/100`` on ``input_fasta``; returns the
    cluster TSV path (reused if it already exists)."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"clusters{threshold}_indexed.tsv")
    tsv_file = prefix + "_cluster.tsv"
    if os.path.exists(tsv_file):
        print(f"Using existing mmseqs clusters at {tsv_file}")
        return tsv_file
    subprocess.run([mmseqs, "easy-cluster", input_fasta, prefix, os.path.join(out_dir, f"tmp{threshold}"),
                    "--min-seq-id", str(threshold / 100.0), "--threads", str(threads)], check=True)
    return tsv_file


def split_for_crossval(input_file: str, output_file: str, mmseqs_dir: str, threshold: int = 40,
                       n_splits: int = 5, split_seed: int = 0, mmseqs: str = "mmseqs",
                       threads: int = 24, cluster_tsv: Optional[str] = None) -> Dict[str, int]:
    """Cluster ``input_file`` and pickle ``{sequence id: fold}`` to ``output_file``.

    ``cluster_tsv``: reuse an mmseqs TSV computed on ``input_file`` itself (raw headers).
    """
    assert os.path.exists(input_file), input_file
    if cluster_tsv is None:
        os.makedirs(mmseqs_dir, exist_ok=True)
        fasta_ids = write_indexed_fasta(input_file, os.path.join(mmseqs_dir, "all_sequences_indexed.fasta"))
        cluster_tsv = run_mmseqs_clustering(os.path.join(mmseqs_dir, "all_sequences_indexed.fasta"), mmseqs_dir,
                                            threshold, mmseqs=mmseqs, threads=threads)
        reps, members = read_mmseqs_clusters(cluster_tsv)
        reps = [fasta_ids[int(r)] for r in reps]
        members = [fasta_ids[int(m)] for m in members]
    else:
        fasta_ids, _ = load_fasta(input_file)
        reps, members = read_mmseqs_clusters(cluster_tsv)
        reps = mmseqs_to_fasta_ids(reps, fasta_ids)
        members = mmseqs_to_fasta_ids(members, fasta_ids)
    n_duplicated = len(fasta_ids) - len(set(fasta_ids))
    if n_duplicated:
        print(f"Warning: {n_duplicated} FASTA ids occur more than once; the last occurrence sets their fold")
    labelling = assign_folds(reps, members, n_splits=n_splits, seed=split_seed)

    missing = set(fasta_ids) - set(labelling)
    if missing:
        raise ValueError(f"{len(missing)} FASTA ids are absent from {cluster_tsv} (e.g. {sorted(missing)[:3]}); "
                         "was it built from a different FASTA?")
    counts = [0] * n_splits
    for fold in labelling.values():
        counts[fold] += 1
    print(f"Fold sizes (sequences): {counts}")
    with open(output_file, "wb") as out_f:
        pickle.dump(labelling, out_f)
    print(f"Saved {output_file}")
    return labelling


# --------------------------
# Command line
# --------------------------

def run_download(cfg):
    p, d = cfg["paths"], cfg["data_download"]
    return download_all_training_data(uniprot_reference_genomes_dir=p["uniprot_reference_genomes_dir"],
                                      uniprot_data_dir=p["uniprot_data_dir"],
                                      ncbi_directory=p["ncbi_directory"],
                                      ogee_data_dir=p["ogee_data_dir"],
                                      label_directory=p["label_folder"],
                                      should_check_labels=d["should_check_labels"],
                                      all_sequences_folder=p["all_sequences_folder"],
                                      taxids_with_custom_download=d["taxids_with_custom_download"])


def run_labels(cfg, max_workers: int = 12):
    p = cfg["paths"]
    os.makedirs(p["label_folder"], exist_ok=True)
    essentiality_df = pd.read_csv(os.path.join(p["ogee_data_dir"], "gene_essentiality.txt"), sep="\t",
                                  encoding_errors="replace", low_memory=False)
    produce_labels_for_many_fastas(essentiality_df=essentiality_df,
                                   all_fasta_folder=p["all_sequences_folder"],
                                   output_label_folder=p["label_folder"],
                                   output_fasta_folder_noduplicates=p["fasta_folder"],
                                   overwrite=False,
                                   max_workers=max_workers,
                                   max_esmc_length=4096)


def run_split(cfg, split_seed: Optional[int] = None, mmseqs: str = "mmseqs", threads: int = 24,
              cluster_tsv: Optional[str] = None) -> str:
    p, s = cfg["paths"], cfg["split"]
    split_seed = s["seed"] if split_seed is None else split_seed
    if not os.path.exists(p["merged_fasta"]):
        print("Merged fasta doesn't exist. Creating merged fasta...")
        merge_different_fasta_files(p["fasta_folder"], p["merged_fasta"])
    output_file = os.path.join(p["splits_folder"], splits_filename(s["prefix"], s["threshold"], split_seed))
    if os.path.exists(output_file):
        print(f"Split already present at {output_file}")
        return output_file
    split_for_crossval(p["merged_fasta"], output_file, p["mmseqs_folder"], threshold=s["threshold"],
                       n_splits=s["n_splits"], split_seed=split_seed, mmseqs=mmseqs, threads=threads,
                       cluster_tsv=cluster_tsv)
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["download", "labels", "split"])
    parser.add_argument("--data-dir", default=None, help="Root of all data paths (default DATA_ROOT/essentiality)")
    parser.add_argument("--config", default=None, help="Config YAML (default: config.yaml next to this file)")
    parser.add_argument("--split-seed", type=int, default=None, help="Split seed (default: config)")
    parser.add_argument("--mmseqs", default="mmseqs", help="mmseqs2 binary")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--cluster-tsv", default=None, help="Reuse an existing mmseqs *_cluster.tsv")
    parser.add_argument("--max-workers", type=int, default=12, help="Processes for the `labels` step")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    if args.step == "download":
        run_download(cfg)
    elif args.step == "labels":
        run_labels(cfg, max_workers=args.max_workers)
    else:
        run_split(cfg, split_seed=args.split_seed, mmseqs=args.mmseqs, threads=args.threads,
                  cluster_tsv=args.cluster_tsv)


if __name__ == "__main__":
    main()
