count=1
ls -l | grep "^d" | while read folder
do
    folder=${folder##* }
    ls $folder/ | while read name
    do
        echo "$folder/$name" $count
    done
    count=$(($count+1))
done > train.txt
