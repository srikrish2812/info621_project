# Open the original file
with open('./test_data.json', 'r') as infile:    #either output2 or 5 or train_dis0_raw
    data = infile.readlines()

# Modify the data to add a comma after every '}'
modified_data = []
for line in data:
    modified_line = line.replace('}', '},')
    modified_data.append(modified_line)

# Write the modified data back to a new file
with open('./commaadd.json', 'w') as outfile:
    outfile.writelines(modified_data)