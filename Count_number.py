from Bio import SeqIO

count = 0
for record in SeqIO.parse("./data/origin/fPETase.fasta", "fasta"):
    count += 1

print("Number of sequences:", count)
