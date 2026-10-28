# -*- coding: utf-8 -*-
"""Runs TrackEval (HOTA, CLEAR, Identity) inside the separate TrackEval environment; must stay Python 3.8 compatible.

It imports nothing from this package.

    <TrackEval python> _trackeval_runner.py <config.json>
"""
import json
import sys


def main(cfg_path):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    sys.path.insert(0, cfg["repo"])
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
        "PRINT_ONLY_COMBINED": False, "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True, "PLOT_CURVES": False,
        "OUTPUT_EMPTY_CLASSES": True,
    })
    name = cfg.get("dataset_class", "MotChallenge2DBox")
    if name == "PersonPath22":
        dataset_cls = trackeval.datasets.PersonPath22
    elif name == "MotChallenge2DBox":
        dataset_cls = trackeval.datasets.MotChallenge2DBox
    else:
        raise ValueError("unknown dataset_class: %r" % (name,))
    ds_config = dataset_cls.get_default_dataset_config()
    ds_config.update({
        "GT_FOLDER": cfg["gt_folder"], "TRACKERS_FOLDER": cfg["trackers_folder"],
        "BENCHMARK": cfg["benchmark"], "SPLIT_TO_EVAL": cfg["split"],
        "TRACKERS_TO_EVAL": cfg["trackers"], "SEQMAP_FILE": cfg["seqmap"],
        "DO_PREPROC": bool(cfg["do_preproc"]), "PRINT_CONFIG": False, "SKIP_SPLIT_FOL": False,
    })
    evaluator = trackeval.Evaluator(eval_config)
    dataset = dataset_cls(ds_config)
    _res, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                                                    trackeval.metrics.Identity()])
    with open(cfg["status_out"], "w", encoding="utf-8") as fh:
        json.dump({"messages": messages, "trackeval_file": getattr(trackeval, "__file__", None)}, fh)


if __name__ == "__main__":
    main(sys.argv[1])
