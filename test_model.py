import transformers
from transformers import RobertaForMaskedLM
from model.nanobert.modeling_nanobert import NanoBertForSequenceClassification
from model.nanobert.configuration_nanobert import NanoBertConfig

if __name__ == "__main__":

    # config = NanoBertConfig()
    model = RobertaForMaskedLM.from_pretrained("NaturalAntibody/nanoBERT")
    model_nano = NanoBertForSequenceClassification.from_pretrained("/Users/zym/Downloads/Research/Okumura_lab/nanobody/model/nanobert")
    for key in model.state_dict().keys():
        print(key)
    
    print("-"*100)
    for key in model_nano.state_dict().keys():
        print(key)

    # tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "/Users/zym/Downloads/Research/Okumura_lab/nanobody/model/nanobert/",
    #     padding_side="right",
    #     use_fast=True,
    #     trust_remote_code=True,
    # )
    # input_text = "ACD"
    # inputs = tokenizer(input_text, return_tensors="pt")
    # outputs = model(**inputs)

    # print(outputs)