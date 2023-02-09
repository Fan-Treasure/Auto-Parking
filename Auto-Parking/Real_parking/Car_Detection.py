

def run_cmd_Popen_fileno(cmd_string):
    """
    执行cmd命令，并得到执行后的返回值，python调试界面输出返回值
    :param cmd_string: cmd命令，如：'adb devices'
    :return:
    """
    import subprocess

    print('运行cmd指令：{}'.format(cmd_string))
    return subprocess.Popen(cmd_string, shell=True, stdout=None, stderr=None).wait()

run_cmd_Popen_fileno('CUDA_VISIBLE_DEVICES=0')
run_cmd_Popen_fileno('python -u PaddleDetection/tools/infer.py -c PaddleDetection/configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml -o weights=Models/model_final --save_txt hhh --infer_dir pic_new')