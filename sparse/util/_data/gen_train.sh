count=1
ls -l | grep "^d" | while read folder
do
    folder=${folder##* }
    count_1=1
    ls $folder/ | while read name
    do
        if [ $count_1 -le $1 ]
        then
            echo "$folder/$name" $count
        fi
        count_1=$(($count_1 + 1))
    done
    count=$(($count+1))
done > train.txt
