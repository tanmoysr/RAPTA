import os
import pickle
import glob
import pandas as pd
import numpy as np
import configure as config

def save_as_csv(labels, outputs, errors,  filename):
    df = pd.DataFrame({"labels": labels, "prediction": outputs, "errors": errors})
    df.to_csv(filename + "_lpe.csv", index=False)

if __name__ == "__main__":
    if config.running_option == 1:
        # reading processed data
        labels = pickle.load(open(config.label_file, 'rb')).flatten()
        outputs = pickle.load(open(config.output_file, 'rb')).flatten()
        errors = labels - outputs
        sigma = np.std(errors) * 1000  # ns to ps
        mu = np.ndarray.mean(errors) * 1000  # ns to ps
        errors_percentage = (np.std(errors) / max(labels)) * 100
        mse = np.square(np.subtract(labels,outputs)).mean()
        print('Std error {}, mean error {}, mse {}, errors_percentage {}'.format(sigma, mu, mse, errors_percentage))
        filename = config.model_chkpnt_path+config.fp
        save_as_csv(labels, outputs, errors, filename)
    else:
        filelist_labels = sorted(glob.glob(config.label_file_multi))
        filelist_outputs = sorted(glob.glob(config.output_file_multi))
        filelist_train_time = sorted(glob.glob(config.training_time_multi))
        filelist_test_time = sorted(glob.glob(config.testing_time_multi))
        print(filelist_train_time)
        print(filelist_labels)
        print(filelist_outputs)

        fpList=[]
        model_time_list = []
        model_time_test_list = []
        lr_time_list=[]
        mean_list=[]
        sigma_list=[]
        errors_percentage_list = []


        for i in range(len(filelist_labels)):
            # reading processed data
            labels = np.array(pickle.load(open(filelist_labels[i], 'rb'))).flatten()
            outputs = np.array(pickle.load(open(filelist_outputs[i], 'rb'))).flatten()

            errors = labels - outputs
            sigma = np.std(errors)*1000 #ns to ps
            sigma_list.append(sigma)
            mu = np.ndarray.mean(errors)*1000 #ns to ps
            mean_list.append(mu)
            errors_percentage = (np.std(errors)/max(labels))*100
            errors_percentage_list.append(errors_percentage)
            fp = filelist_outputs[i].split('\\')[-1][0:-3]
            fpList.append(fp)
            filename = config.model_chkpnt_path + fp

            if config.timing_option==1:
                model_time = pickle.load(open(filelist_train_time[i], 'rb'))
                model_time_list.append(model_time)
                model_time_test = pickle.load(open(filelist_test_time[i], 'rb'))
                model_time_test_list.append(model_time_test)
            save_as_csv(labels, outputs, errors, filename)
            print('Finished files set {}'.format(fp))

        # saving in csv
        if config.timing_option == 1:
            df = pd.DataFrame({"Exp": fpList, "Training Time (sec)": model_time_list, "Testing Time (sec)": model_time_test_list,
                               "Error_percentage": errors_percentage_list, "Mean(ps)": mean_list, "Std_error(ps)": sigma_list})
            df.to_csv(config.model_chkpnt_path + config.appl_bench + "_Full_Results.csv", index=False)
        else:
            df = pd.DataFrame(
                {"Exp": fpList, "Mean(ps)": mean_list,
                 "Std_error(ps)": sigma_list})
            df.to_csv(config.model_chkpnt_path + config.appl_bench + "_Full_Results_conventional.csv", index=False)
