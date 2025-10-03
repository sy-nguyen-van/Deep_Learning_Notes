rm -f assignment1.zip 
zip -r assignment1.zip . -x "*lib/datasets*" "*.ipynb_checkpoints*" "*collectSubmission.sh" "*requirements.txt"
