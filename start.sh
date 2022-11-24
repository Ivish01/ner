#!/bin/bash -x
set -x
export BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $BASE_PATH
export LOGDIR=${LOGDIR:=.}
export PYTHONPATH="$BASE_PATH/models/libraries:$BASE_PATH:$PYTHONPATH"
touch ${LOGDIR}/ech.log 
echo "ENVIRONMENT=${ENVIRONMENT:=production}" >> ${LOGDIR}/ech.log
echo "GIT_BRANCH=${GIT_BRANCH:=master}" >> ${LOGDIR}/ech.log
export ENVIRONMENT REDIS_URL POSTGRES_URL ARANGO_URL
#git clean -f -d
#git checkout ${GIT_BRANCH}
#git reset --hard HEAD
#git pull --prune origin ${GIT_BRANCH}
#git checkout ${GIT_BRANCH}
#git log -1 --pretty=format:"%H:%aI" > dashboard.sha
# Ensure requirements are met
#set -e
#pip install -r requirements.txt
#set +e
# Startup the settings
python3.8 app.py $* 2>&1 | tee ${LOGDIR}/start.log
pid=$!
#status=${PIPESTATUS[0]}
#if [[ $status != 0 ]];
#then
#    cat ${LOGDIR}/start.log
#    exit $status
#fi

#export LOG_FILE=${LOGDIR}/dashboard.log
#if [[ -f dashboard.pid ]]; then
#    kill -9 `cat dashboard.pid`
#    sleep 10
#fi
#rm -f dashboard.pid
#nohup waitress-serve --port=5000 --url-scheme=http --threads=${API_THREADS} app:app 1>> ${LOGDIR}/dashboard_stdout.log 2>> ${LOGDIR}/dashboard_stderr.log &
#pid=$!
#echo "DASH PID=${pid}" | tee -a ${LOGDIR}/ech.log
#echo $pid > dashboard.pid
#sleep 10

echo "Done with deploy" >&2
jobs -l | grep $pid
while [[ -n `jobs | grep $pid | grep -v Exit` ]]; do 
    jobs -l | grep $pid
    sleep 10; 
done


