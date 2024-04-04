mkdir -p /home/markusheimerl/xvit-415020_dolma/
gcsfuse xvit-415020_dolma /home/markusheimerl/xvit-415020_dolma/
sleep 5
cp /home/markusheimerl/xvit-415020_dolma/books-0000.json.gz /home/markusheimerl/transformer/dolma/ &
cp /home/markusheimerl/xvit-415020_dolma/books-0001.json.gz /home/markusheimerl/transformer/dolma/ &
cp /home/markusheimerl/xvit-415020_dolma/books-0002.json.gz /home/markusheimerl/transformer/dolma/
cp /home/markusheimerl/xvit-415020_dolma/en_simple_wiki_v0-0000.json.gz /home/markusheimerl/transformer/dolma/ &
cp /home/markusheimerl/xvit-415020_dolma/en_simple_wiki_v0-0001.json.gz /home/markusheimerl/transformer/dolma/
sleep 20
fusermount -u "/home/markusheimerl/xvit-415020_dolma/"