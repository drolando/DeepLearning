FOLDER="no_mean"
NET_FOLDER="../../demos/mine/no_mean"

while getopts on option
do
    case "${option}"
    in
        o) FOLDER=${OPTARG};;
        n) NET_FOLDER=${OPTARG};;
    esac
done

for i in {1..6}
do
    echo "Iteration: $i"
    START=$(date +%s.%N)
    echo "Scaling.."
    if [ ! -f "$FOLDER/plibsvm_$i.train" ]; then
        ./svm-scale -s tmp.txt $NET_FOLDER/plibsvm_$i.train > "$FOLDER/plibsvm_$i.train"
    fi
    if [ ! -f "$FOLDER/plibsvm_$i.val" ]; then
        ./svm-scale -r tmp.txt $NET_FOLDER/plibsvm_$i.val > "$FOLDER/plibsvm_$i.val"
    fi
    echo "Training.."
    ./svm-train -b 1 -t 1 -d 3 -g 0.0078125 "$FOLDER/plibsvm_$i.train" "$FOLDER/plibsvm_$i.model"
    echo "Predict.."
    ./svm-predict -b 1 "$FOLDER/plibsvm_$i.val" "$FOLDER/plibsvm_$i.model" "$FOLDER/plibsvm_$i.out"
    python saved/precision.py "$FOLDER/plibsvm_$i.val" "$FOLDER/plibsvm_$i.out" &> "$FOLDER/plibsvm_$i.prec"
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo "Duration iteration $i: $DIFF"
done

