count=1
ls -l | grep "^d" | while read folder
do
    folder=${folder##* }
    count_1=0
    ls $folder/ | while read name
    do
        tmp=$(grep "$name" train.txt)
        if [ $? -eq 1 ]
        then
            if [ $count_1 -ge $1 ] && [ $count_1 -le $2 ]
            then
                echo "$folder/$name" $count
            fi
            count_1=$(($count_1 + 1))
        fi
    done
    count=$(($count+1))
done > val.txt
