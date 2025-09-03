def read_fasta(file_path):
    """
    Reads a FASTA file and extracts sequences into a dictionary.
    (Same as your original implementation)

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        dict: A dictionary where keys are sequence IDs and values are sequences.
    """
    with open(file_path, 'r') as file:
        sequences = {}
        sequence_id = None
        sequence_data = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):  # Header line
                if sequence_id:
                    sequences[sequence_id] = ''.join(sequence_data)
                sequence_id = line[1:].split()[0]  # Extract sequence ID
                sequence_data = []
            else:
                sequence_data.append(line)
        if sequence_id:
            sequences[sequence_id] = ''.join(sequence_data)
    return sequences