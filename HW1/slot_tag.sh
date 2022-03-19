wget https://www.dropbox.com/s/qko2m2ohqr7rtw2/Slot-e472c610.pt?dl=1 -O Slot-e472c610.pt

python3.8 test_slot.py --test_file "${1}" --ckpt_path Slot-e472c610.pt --pred_file "${2}"

# Local
# bash slot_tag.sh .\data\slot\test.json slot.resutl.csv
# python test_slot.py --test_file "${1}" --ckpt_path ckpt/slot/Slot-e472c610.pt --pred_file "${2}"