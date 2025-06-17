import csv
from datasets import load_dataset


with open("../../datasets/raw/TED2020.en-ko.en", "r") as f_en, \
    open("../../datasets/raw/TED2020.en-ko.ko", "r") as f_ko, \
    open("../../datasets/csv/TED2020_en-ko.csv", "w", newline="") as f_out:
        
        writer = csv.writer(f_out)
        writer.writerow(["en", "ko"]) # header
        
        seen_pairs = set()
        
        for en_line, ko_line in zip(f_en, f_ko):
            en_line = en_line.strip()
            ko_line = ko_line.strip()
            
            pair = (en_line, ko_line)
            
            if pair not in seen_pairs:
                writer.writerow([en_line, ko_line])
                seen_pairs.add(pair)

datasets = load_dataset("csv", data_files="../../datasets/csv/TED2020_en-ko.csv", split="train")

datasets.save_to_disk("../../datasets/dataset/TED2020_en-ko")

datasets.push_to_hub(repo_id="Jooinjang/TED2020_en-ko", private=True) # replace with your repo id