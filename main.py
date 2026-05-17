import json
import importlib
import torch
from option import get_option
from solver import Solver
from tester1 import Tester
from utils import LogWritter
import glob


# def main():
#     opt = get_option()
#     torch.manual_seed(opt.seed)

#     module = importlib.import_module("model.{}".format(opt.model))
#     logger = LogWritter(opt)

#     if not opt.test_only:
#         msg = json.dumps(vars(opt), indent=4)
#         print(msg)
#         logger.update_txt(msg + '\n', mode='w')

#     if opt.test_only:
#         solver = Solver(module, opt, 0, True)
#         solver.evaluate()
#     else:
#         for e in range(0, 5):
#             solver = Solver(module, opt, e, True)
#             solver.fit()
def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    module = importlib.import_module("model.{}".format(opt.model))
    logger = LogWritter(opt)

    if not opt.test_only:
        msg = json.dumps(vars(opt), indent=4)
        print(msg)
        logger.update_txt(msg + '\n', mode='w')
        for e in range(0, 1):
            solver = Solver(module, opt, e, train=True)
            solver.fit()
    else:
        # 使用 Tester 类
        from tester1 import Tester
        tester = Tester(module, opt)
        tester.test()


if __name__ == "__main__":
    main()
