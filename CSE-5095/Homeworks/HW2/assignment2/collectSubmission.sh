rm -f assignment2.zip 
zip -r assignment2.zip . -x "*lib/datasets*" "*.ipynb_checkpoints*" "*.idea" "*collectSubmission.sh" "*requirements.txt"
