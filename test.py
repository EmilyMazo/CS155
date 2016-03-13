# The following code reads each sonnet from a text file.
# Each sonnet is input as a single element into sonnet_list.
# Each line of each sonnet is prepended with <s> and appended with </s>.
# There is currently no other preprocessing.

def tokenizeSequences(filename):
    training = []
    training_temp = []
    sonnets = open(filename, "r")
    sonnet_list = []
    #sonnet_list_temp = []
    observations = {}
    counter = 0
    for line in sonnets:
        line = line.strip()
        line = line.split(' ')
        if len(line) == 1:
            if line == ['']:
                if counter <= 0:
                    continue
                else:
                    break
            counter += 1
            #sonnet_list.append("<sonnet>")
            #sonnet_list.append(sonnet_list_temp)
            training.append(training_temp)
            #sonnet_list_temp = []
            continue
        new_line = []
        for l in line:
            new_line.append(l)
        for l in new_line:
            if l not in observations:
                observations[l] = 1
            #sonnet_list_temp.append(l)
            sonnet_list.append(l)
            training_temp.append((l, ''))
        #sonnet_list.append("</sonnet>")
    training.append(training_temp)
    #sonnet_list.remove([])
    sonnets.close()
    print training
    return training, sonnet_list, observations.keys(), len(observations.keys())

# This version adds start-of-line, end-of-line, start-of-sonnet, and end-of-sonnet tags
def tokenizeSequences(filename):
    training = []
    training_temp = []
    sonnets = open(filename, "r")
    sonnet_list = []
    #sonnet_list_temp = []
    observations = {}
    counter = 0
    for line in sonnets:
        line = line.strip()
        line = line.split(' ')
        if len(line) == 1:
            if line == ['']:
                if counter <= 2:
                    continue
                else:
                    break
            counter += 1
            #sonnet_list.append("<sonnet>")
            #sonnet_list.append(sonnet_list_temp)
            training.append(training_temp)
            training_temp = [('startofsonnet', '')]
            #sonnet_list_temp = []
            continue
        line.append("endofline")
        new_line = []
        new_line.append("startofline")
        for l in line:
            new_line.append(l)
        for l in new_line:
            if l not in observations:
                observations[l] = 1
            #sonnet_list_temp.append(l)
            sonnet_list.append(l)
            training_temp.append((l, ''))
        #sonnet_list.append("</sonnet>")
    training_temp.append(("endofsonnet", ''))
    training.append(training_temp)
    training.remove([])
    #sonnet_list.remove([])
    sonnets.close()
    observations['startofsonnet'] = 1
    observations['startofline'] = 1
    observations['endofsonnet'] = 1
    observations['endofline'] = 1
    print training
    return training, sonnet_list, observations.keys(), len(observations.keys())    
    
# This version adds start-of-line, end-of-line, start-of-sonnet, and end-of-sonnet tags
# and also removes punctuation
def tokenizeSequences(filename):
    training = []
    training_temp = []
    sonnets = open(filename, "r")
    sonnet_list = []
    #sonnet_list_temp = []
    observations = {}
    counter = 0
    for line in sonnets:
        line = line.strip()
        line = line.split(' ')
        line =  [word.strip(".,?;:()").lower() for word in line]
        if len(line) == 1:
            if line == ['']:
                if counter <= 2:
                    continue
                else:
                    break
            counter += 1
            #sonnet_list.append("<sonnet>")
            #sonnet_list.append(sonnet_list_temp)
            training.append(training_temp)
            training_temp = [('startofsonnet', '')]
            #sonnet_list_temp = []
            continue
        line.append("endofline")
        new_line = []
        new_line.append("startofline")
        for l in line:
            new_line.append(l)
        for l in new_line:
            if l not in observations:
                observations[l] = 1
            #sonnet_list_temp.append(l)
            sonnet_list.append(l)
            training_temp.append((l, ''))
        #sonnet_list.append("</sonnet>")
    training_temp.append(("endofsonnet", ''))
    training.append(training_temp)
    training.remove([])
    #sonnet_list.remove([])
    sonnets.close()
    observations['startofsonnet'] = 1
    observations['startofline'] = 1
    observations['endofsonnet'] = 1
    observations['endofline'] = 1
    print training
    return training, sonnet_list, observations.keys(), len(observations.keys())    

 # This version adds start-of-line, end-of-line, start-of-sonnet, and end-of-sonnet tags
# and also makes punctuation into new tokens
import re

def tokenizeSequences(filename):
    training = []
    training_temp = []
    sonnets = open(filename, "r")
    sonnet_list = []
    #sonnet_list_temp = []
    observations = {}
    counter = 0
    for line in sonnets:
        line = line.strip()
        line = line.strip("(")
        line = line.strip(")")
        line = re.split('\s|[?.,!:;]', line)
        if len(line) == 1:
            if line == ['']:
                if counter <= 0:
                    continue
                else:
                    break
            counter += 1
            #sonnet_list.append("<sonnet>")
            #sonnet_list.append(sonnet_list_temp)
            training.append(training_temp)
            training_temp = [('startofsonnet', '')]
            #sonnet_list_temp = []
            continue
        line.append("endofline")
        new_line = []
        new_line.append("startofline")
        for l in line:
            new_line.append(l)
        for l in new_line:
            if l not in observations:
                observations[l] = 1
            #sonnet_list_temp.append(l)
            sonnet_list.append(l)
            if l == "''":
                continue
            training_temp.append((l, ''))
        #sonnet_list.append("</sonnet>")
    training_temp.append(("endofsonnet", ''))
    training.append(training_temp)
    training.remove([])
    #sonnet_list.remove([])
    sonnets.close()
    observations['startofsonnet'] = 1
    observations['startofline'] = 1
    observations['endofsonnet'] = 1
    observations['endofline'] = 1
    observations['.'] = 1
    observations[','] = 1
    observations['?'] = 1
    observations['!'] = 1
    observations[':'] = 1
    observations[';'] = 1
    print training
    return training, sonnet_list, observations.keys(), len(observations.keys())    

