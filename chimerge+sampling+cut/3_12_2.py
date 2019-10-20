from time import ctime


def read(file):
    '''read raw date from a file '''
    Instances = []
    fp = open(file, 'r')
    for line in fp:
        line = line.strip('\n')  # discard '\n'
        if line != '':
            Instances.append(line.split(','))
    fp.close()
    return (Instances)


def split(Instances, i):
    ''' Split the 4 attibutes, collect the data of the ith attributs, i=0,1,2,3
        Return a list like [['0.2', 'Iris-setosa'], ['0.2', 'Iris-setosa'],...]'''
    log = []
    for r in Instances:
        log.append([r[i], r[4]])
    return (log)


def count(log):
    '''Count the number of the same record
       Return a list like [['4.3', 'Iris-setosa', 1], ['4.4', 'Iris-setosa', 3],...]'''
    log_cnt = []
    log.sort(key=lambda log: log[0])
    i = 0
    while (i < len(log)):
        cnt = log.count(log[i])  # count the number of the same record
        record = log[i][:]
        record.append(cnt)  # the return value of append is None
        log_cnt.append(record)
        i += cnt  # count the next diferent item
    return (log_cnt)


def build(log_cnt):
    '''Build a structure (a list of truples) that ChiMerge algorithm works properly on it '''
    log_dic = {}
    for record in log_cnt:
        if record[0] not in log_dic.keys():
            log_dic[record[0]] = [0, 0, 0]
        if record[1] == 'Iris-setosa':
            log_dic[record[0]][0] = record[2]
        elif record[1] == 'Iris-versicolor':
            log_dic[record[0]][1] = record[2]
        elif record[1] == 'Iris-virginica':
            log_dic[record[0]][2] = record[2]
        else:
            raise TypeError("Data Exception")
    log_truple = sorted(log_dic.items())
    return (log_truple)


def collect(Instances, i):
    ''' collect data for discretization '''
    log = split(Instances, i)
    log_cnt = count(log)
    log_tuple = build(log_cnt)
    return (log_tuple)


def combine(a, b):
    '''  a=('4.4', [3, 1, 0]), b=('4.5', [1, 0, 2])
         combine(a,b)=('4.4', [4, 1, 2])  '''
    c = a[:]  # c[0]=a[0]
    for i in range(len(a[1])):
        c[1][i] += b[1][i]
    return (c)


def chi2(A):
    ''' Compute the Chi-Square value '''
    m = len(A);
    k = len(A[0])
    R = []
    for i in range(m):
        sum = 0
        for j in range(k):
            sum += A[i][j]
        R.append(sum)
    C = []
    for j in range(k):
        sum = 0
        for i in range(m):
            sum += A[i][j]
        C.append(sum)
    N = 0
    for ele in C:
        N += ele
    res = 0
    for i in range(m):
        for j in range(k):
            Eij = R[i] * C[j] / N
            if Eij != 0:
                res = res + (A[i][j] - Eij) ** 2 / Eij
    return res


def ChiMerge(log_tuple, max_interval):
    ''' ChiMerge algorithm  '''
    ''' Return split points '''
    num_interval = len(log_tuple)
    while (num_interval > max_interval):
        num_pair = num_interval - 1
        chi_values = []
        for i in range(num_pair):
            arr = [log_tuple[i][1], log_tuple[i + 1][1]]
            chi_values.append(chi2(arr))
        min_chi = min(chi_values)  # get the minimum chi value
        for i in range(num_pair - 1, -1, -1):  # treat from the last one
            if chi_values[i] == min_chi:
                log_tuple[i] = combine(log_tuple[i], log_tuple[i + 1])  # combine the two adjacent intervals
                log_tuple[i + 1] = 'Merged'
        while ('Merged' in log_tuple):  # remove the merged record
            log_tuple.remove('Merged')
        num_interval = len(log_tuple)
    split_points = [record[0] for record in log_tuple]
    return (split_points)


def discrete(path):
    ''' ChiMerege discretization of the Iris plants database '''
    Instances = read(path)
    max_interval = 6
    num_log = 4
    for i in range(num_log):
        log_tuple = collect(Instances, i)  # collect data for discretization
        split_points = ChiMerge(log_tuple, max_interval)  # discretize data using ChiMerge algorithm
        print(split_points)


if __name__ == '__main__':
    print('Start: ' + ctime())
    discrete('c:\\Users\\Tianh\\Desktop\\iris_data.csv')
    print('End: ' + ctime())