def load_all_go_terms(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    mf_terms, bp_terms, cc_terms = [], [], []

    for i, line in enumerate(lines):
        if line.startswith("### GO-terms (molecular_function)"):
            mf_terms = lines[i+1].strip().split('\t')
        elif line.startswith("### GO-terms (biological_process)"):
            bp_terms = lines[i+1].strip().split('\t')
        elif line.startswith("### GO-terms (cellular_component)"):
            cc_terms = lines[i+1].strip().split('\t')
        elif line.startswith("### PDB-chain"):
            annotation_start = i + 1
            break

    go_map = {}

    for line in lines[annotation_start:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        while len(parts) < 4:
            parts.append('')
        
        pdb_chain = parts[0]
        mf_list = parts[1].split(',') if parts[1] else []
        bp_list = parts[2].split(',') if parts[2] else []
        cc_list = parts[3].split(',') if parts[3] else []

        mf_vec = [1 if go in mf_list else 0 for go in mf_terms]
        bp_vec = [1 if go in bp_list else 0 for go in bp_terms]
        cc_vec = [1 if go in cc_list else 0 for go in cc_terms]

        go_map[pdb_chain.lower()] = {
            'mf': mf_vec,
            'bp': bp_vec,
            'cc': cc_vec
        }

    return go_map



def load_all_ec_numbers(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Step 1: Parse GO term lists from header
    ec_numbers = []

    for i, line in enumerate(lines):
        if line.startswith("### EC-numbers"):
            print(" Line started with EC")
            ec_numbers = lines[i+1].strip().split('\t')
        elif line.startswith("### PDB-chain"):
            print(" Line started with PDB")
            annotation_start = i + 1
            break

    # Step 2: Extract and parse annotation lines

    ec_map = {}
    for line in lines[annotation_start:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        pdb_chain = parts[0]
        ec_list = parts[1].split(',') if len(parts) > 1 and parts[1] else []

        # Step 3: Create binary vector for EC numbers
        ec_vec = [1 if ec in ec_list else 0 for ec in ec_numbers]

        ec_map[pdb_chain.lower()] = {
            'ec':ec_vec
        }

    # Step 4: Convert to DataFrame
    return ec_map
