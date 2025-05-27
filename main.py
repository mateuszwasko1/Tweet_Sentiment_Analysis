from project_name import BaselineModel

if __name__ == '__main__':
    baseline = BaselineModel()

    baseline_metrics =baseline.pipeline()
    print(baseline.best_parameters)
    print(baseline_metrics)
