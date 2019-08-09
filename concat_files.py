import sys
def concat_files_zip(file1, file2, output):
    with open(output, 'w', encoding='utf-8') as fout:
        with open(file1, 'r', encoding='utf-8') as fin1, open(file2, 'r', encoding='utf-8') as fin2:
            for line1, line2 in zip(fin1, fin2):
                fout.write(line1.strip() + ' <s> ' + line2.strip()+'\n')
                fout.flush()

concat_files_zip(sys.argv[1], sys.argv[2], sys.argv[3])


