import glob
import os
import shutil
from collections import OrderedDict
import torch

class Saver:
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(self.args.exp_path, args.dataset, args.checkname)
        self.runs = sorted([int(path_exp.split("_")[-1]) for path_exp in glob.glob(os.path.join(self.directory, "experiment_*"))])
        run_id = self.runs[-1] + 1 if self.runs else 0
        self.experiment_dir = os.path.join(self.directory, "experiment_{}".format(str(run_id)))
        print("Experiment_dir: ", self.experiment_dir)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        with open(self.experiment_dir + "/args.txt", "w") as text_file:
            text_file.write(str(args))

    def save_checkpoint(
        self, state, is_best, filename="checkpoint.pth.tar", generator_state=None, uncertainty_estimator_state=None
    ):
        """Saves checkpoint to disk"""
        filename_generator = os.path.join(self.experiment_dir, "generator_" + filename)
        filename_uncertainty_estimator = os.path.join(self.experiment_dir, "uncertainty_estimator_" + filename)
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        torch.save(generator_state, filename_generator)
        torch.save(uncertainty_estimator_state, filename_uncertainty_estimator)
        if is_best:
            # best_pred = state["best_pred"]
            # with open(os.path.join(self.experiment_dir, "best_pred.txt"), "w") as f:
            #     f.write(str(best_pred))
            if self.runs:
                # previous_miou = [0.0]
                # for run in self.runs:
                    # run_id = run
                    # path = os.path.join(
                    #     self.directory, "experiment_{}".format(str(run_id)), "best_pred.txt",
                    # )
                    # if os.path.exists(path):
                    #     with open(path, "r") as f:
                    #         miou = float(f.readline())
                    #         previous_miou.append(miou)
                    # else:
                    #     continue
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.experiment_dir, str(state["epoch"]) + "_model.pth.tar"
                    ),
                )
                shutil.copyfile(
                    filename_generator,
                    os.path.join(
                        self.experiment_dir, str(state["epoch"]) + "_generator.pth.tar"
                    ),
                )
                shutil.copyfile(
                    filename_uncertainty_estimator,
                    os.path.join(
                        self.experiment_dir, str(state["epoch"]) + "_uncertainty_estimator.pth.tar"
                    ),
                )
            else:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.experiment_dir, str(state["epoch"]) + "_model.pth.tar"
                    ),
                )
                shutil.copyfile(
                    filename_uncertainty_estimator,
                    os.path.join(
                        self.experiment_dir, str(state["epoch"]) + "_uncertainty_estimator.pth.tar"
                    ),
                )

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, "parameters.txt")
        log_file = open(logfile, "w")
        p = OrderedDict()
        p["datset"] = self.args.dataset
        p["lr"] = self.args.lr
        p["lr_scheduler"] = self.args.lr_scheduler
        p["loss_type"] = self.args.loss_type
        p["epoch"] = self.args.epochs
        p["epoch"] = self.args.iter

        for key, val in p.items():
            log_file.write(key + ":" + str(val) + "\n")
        log_file.close()
