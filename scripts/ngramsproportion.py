import re
import csv

def clean_string_to_list(str_):
    wordsnoquote = str_.replace("'","").replace('"','')
    wordsnocomma = re.sub(r'(?!(([^"]*"){2})*[^"]*$),', '', wordsnoquote)
    listofwords = wordsnocomma.replace("[","").replace("]","").replace(" ", "").split(",")
    return list(filter(None, listofwords))
    
# load up data
names = ['5055','5560','6065','6570','7075','7580','8085','8590','9095','9500']

filedict = {}
print("Compute proportion")
for name in names:
    gram3 = 0
    gram4 = 0
    gram5 = 0
    gramno = 0
    print("Ngrams with score : " + str(name))
    with open('/mnt/storage01/milliet/data/ngrams/clean-ngrams-score-'+name+'.csv', 'r') as csvfile:
        #lines = csvfile.readlines()
        lenfile = 1#len(csvfile)
        i=0
        for line in csvfile:
            if i%10000==0:
                print("Line " + str(i) + " / " + str(lenfile), end="\r")
            dataline = line.split('\sep')
            words = dataline[0]
            listofwords = clean_string_to_list(words)
            if len(listofwords)==3:
                gram3+=1
            elif len(listofwords)==4:
                gram4+=1
            elif len(listofwords)==5:
                gram5+=1
            else:
                gramno+=1
            i+=1
    print("---------------------------")
    print("3grams : " + str(gram3))
    print("4grams : " + str(gram4))
    print("5grams : " + str(gram5))
    print("No 3-4-5grams : " + str(gramno))
    filedict[name] = [gram3, gram4, gram5]   

print("Write file")
with open('/mnt/storage01/milliet/embedding/clean-ngramsProportion.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in filedict.items():
       writer.writerow([key, value])
print("End of script")