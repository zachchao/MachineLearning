


# Result should be [0, 1]
x = [0.2, 0.5]

# Result should be [1, 0]
x = [0.5, 0.2]



def one_hot(x):
	return [i // max(x) for i in x]

print(one_hot(x))