import os, sys, optparse, warnings
from Bio.Seq import Seq
import h5py, multiprocessing
import numpy as np
import theano
import keras
from keras.models import load_model
from encode import encodeSeq
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def pred(ID) :
    code_p = code[ID]
    code_n = codeR[ID]
    head = seqname[ID]
    
    length = len(code_p)
    
    if length < 300 :
      model = model_dict[0.15]
      null = null_dict[0.15]
    elif length < 500 and length >= 300 :
      model = model_dict[0.3]
      null = null_dict[0.3]
    elif length < 1000 and length >= 500 :
      model = model_dict[0.5]
      null = null_dict[0.5]
    else :
      model = model_dict[1]
      null = null_dict[1]
    print("Starting predict " + str(ID) +" DNA_seq")
    score = model.predict([np.array([code_p]), np.array([code_n])], batch_size=1)
    pvalue = sum([x>score for x in null])/len(null)
    writef = predF.write('\t'.join([head, str(length), str(float(score)), str(float(pvalue))])+'\n')
    flushf = predF.flush()
    
    return [head, float(score), float(pvalue)]

if __name__ == "__main__":
    prog_base = os.path.split(sys.argv[0])[1]
    parser = optparse.OptionParser()
    parser.add_option("-i", "--in", action = "store", type = "string", dest = "filename", 
                    help = "input fasta file")
    parser.add_option("-m", "--mod", action = "store", type = "string", dest = "model_dir",
                                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), 
                    help = "model directory (default ./models)")
    parser.add_option("-o", "--out", action = "store", type = "string", dest = "output_dir",
                                        default='./test', help = "output directory")
    parser.add_option("-l", "--len", action = "store", type = "int", dest = "cutoff_len",
                                        default=300, help = "predict only for sequence >= L bp (default 1)")
    parser.add_option("-c", "--core", action = "store", type = "int", dest = "core_num",
                                        default=1, help = "number of parallel cores (default 1)")
    cur_filename = os.path.split(sys.argv[0])[1]
    (option, args) = parser.parse_args()
    if option.filename is None:
            sys.stderr.write("-------------------------------------------------------------------------------\n" +
            cur_filename + " appear ERROR: missing some required command argument" + 
                        "\n-------------------------------------------------------------------------------\n")
            parser.print_help()
            sys.exit(0)
    else :
        print("\nStep 1: set up the parameters\n")
        print("---------------------------------------------------------------------")
        print("|Initialize required argument:"+
        "\n|input fasta file is     : " + option.filename +
        "\n|model directory is      : " + option.model_dir +
        "\n|output directory is     : " + option.output_dir +
        "\n|cutoff_len is           : " + str(option.cutoff_len) +
        "\n|core number is          : " + str(option.core_num) )
        print("---------------------------------------------------------------------")

    filename = option.filename
    model_dir = option.model_dir
    output_dir = option.output_dir
    cutoff_len = option.cutoff_len
    cutoff_lenk = str(cutoff_len)
    core_num = option.core_num

    print("Step 2: Loading model\n")
    model_dict = {}
    null_dict = {}
    for cut_len in [0.15, 0.3, 0.5, 1]:
        model = 'model_siamese_varlen_'+ str(cut_len) +'k'
        model_name = [ f for f in os.listdir(model_dir) if model in f and f.endswith(".h5")][0]
        model_dict[cut_len] = load_model(os.path.join(model_dir, model_name))
        Y_pred_filename = [f for f in os.listdir(model_dir) if model in f and "Y_pred" in f][0]
        with open(os.path.join(model_dir, Y_pred_filename)) as f:
            lines = [line.split() for line in f][0]
            Y_pred = [float(x) for x in lines]
        Y_real_filename = [f for f in os.listdir(model_dir) if model in f and "Y_true" in f][0]   
        with open(os.path.join(model_dir, Y_real_filename)) as f:
            lines = [line.split()[0] for line in f]
            Y_real = [float(x) for x in lines]
        null_dict[cut_len] = Y_pred[: Y_real.index(1)]

    print("Step 3: encode sequence and predict score")
    outfile = os.path.join(output_dir, os.path.basename(filename)+'_gt'+str(cutoff_len)+'bp_pred.txt')
    predF = open(outfile, 'w')
    writef = predF.write('\t'.join(['name', 'len', 'score', 'pvalue'])+'\n')
    predF.close()
    predF = open(outfile, 'a')

    with open(filename, 'r') as f:
        code = []
        codeR = []
        seqname = []
        head = ''
        DNA_seq = ''
        DNA_number = 0
        flag = 1
        error_N = 0
        for line in f:
            if line[0] == '>':
                if flag == 0:
                    error_N = DNA_seq.count('N')/len(DNA_seq)
                    if (error_N <=0.3 and len(DNA_seq) >= cutoff_len):
                        code.append(encodeSeq(DNA_seq))
                        codeR.append(encodeSeq(Seq(DNA_seq).reverse_complement()))
                        seqname.append(head)
                    else:
                        print("The " + str(DNA_number) +'-th is ignored\n')
                    flag = 1
                DNA_number += 1
                print ("Start encoding " + str(DNA_number) + "-th DNA_sequence\n")
                head = line.strip()[1:]
                DNA_seq = ''
            else:
                flag = 0
                DNA_seq += line.strip()

        error_N = DNA_seq.count('N')/len(DNA_seq)
        if (error_N <=0.3 and len(DNA_seq) >= cutoff_len):
            code.append(encodeSeq(DNA_seq))
            codeR.append(encodeSeq(Seq(DNA_seq).reverse_complement()))
            seqname.append(head)
        
        for i in range(0,len(code)):
            pred(i)
            #pool = multiprocessing.Pool(core_num)
            #head, score, pvalue = zip(*pool.map(pred, range(0, len(code))))
            #pool.close()
    predF.close()

    print("Predict finished!")