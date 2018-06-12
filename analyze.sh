for x in *.py;
do pylint3 $x | grep -E "\* Module| rated at "
done
