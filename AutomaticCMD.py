import os
layout = os.listdir('layouts')
algorithms = ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']
functions = ['scoreEvaluationFunction','betterEvaluationFunction']
f = open("script.sh", "w")
f.write('python reset_results.py \n')
fi = open('title.txt', "w+")
for func in functions:
    for level in layout:
        for al in algorithms:
            fi.writelines(f'The results of map {level[:-4]} with {al} implementation. \n')
            command = f'python pacman.py -l {level[:-4]} -p {al} -a depth=2,evalFn={func} -q \n'
            f.write(command)
f.close()
fi.close()