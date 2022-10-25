pkill python
mkdir bak
mv *.pkl ./bak

exitfn () {
	trap SIGINT
	mv ./bak/* .
	rm -rf ./bak
}

trap "exitfn" INT 

STAGING=1 python3 -u cl_runner.py

trap SIGINT
