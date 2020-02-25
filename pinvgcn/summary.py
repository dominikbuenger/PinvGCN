
import os
import numpy as np
from contextlib import redirect_stdout


def print_results(accuracies, setup_times, training_times, *dictionaries, 
                      architecture=None, dataset=None, print_all=False, **kwargs):
    
    
    def print_single(varname, var, unit=''):
        if var is not None:
            print(" - {}: {} {}".format(varname, np.mean(var), unit))
    def print_multi(varname, var, unit=''):
        if var is not None:
            if np.size(var) > 1:
                print(" - Average {}: {} {} +- {}".format(varname, np.mean(var), unit, np.std(var)))
            else:
                print(" - Constant {}: {} {}".format(varname, var, unit))
                
    N = len(accuracies)
    if N == 0:
        return
    if N == 1:
        run_string = "a single run"
        p = print_single
    else:
        run_string = "{} runs".format(N)
        p = print_multi
        
    if architecture is not None:
        print("Summary of {} with architecture \"{}\"".format(run_string, architecture)
              + (":" if dataset is None else " on dataset \"{}\":".format(dataset)))
    
    p("Accuracy", 100*np.array(accuracies), '%')
    p("Setup Time", setup_times, 's')
    p("Training Time", training_times, 's') 
    p("Total Time", np.add(setup_times, training_times), 's')
    
    if len(dictionaries) > 0 or len(kwargs) > 0:
        print()
        dictionaries = list(dictionaries)
        dictionaries.append(kwargs)
        for d in dictionaries:
            if not isinstance(d, dict):
                d = d.__dict__
            for key, val in d.items():
                print(' - {} = {}'.format(key, val))
    
    if N > 1 and print_all:
        print()
        print("Individual run results:")
        if isinstance(setup_times, list):
            print(" Run   SetupTime  TrainingTime  Accuracy")
            for run in range(N):
                print(" {: 3d}  {: 12.4f}  {: 12.4f}  {:.4f}".format(run+1, setup_times[run], training_times[run], 100*accuracies[run]))
        else:
            print(" Run  TrainingTime  Accuracy")
            for run in range(N):
                print(" {: 3d}  {: 12.4f}  {:.4f}".format(run+1, training_times[run], 100*accuracies[run]))
            

def save_results(dir, dataset, architecture, *args, **kwargs):
    dir = os.path.join(dir, dataset)
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, architecture + '.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            print_results(*args, architecture=architecture, dataset=dataset, print_all=True, **kwargs)
    print('Results saved to file {}'.format(filename))