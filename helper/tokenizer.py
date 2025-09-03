from transformers import BertModel, BertTokenizer,AutoTokenizer



def store_tokenizer(tokenizer,path):
    tokenizer.save_pretrained(path)
def load_new_tokenizer(model_name = "Rostlab/prot_bert"):
    # Pretrained ProteinBERT model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    return tokenizer
def load_pretrained_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)