pushd .

cd D:\GitHubSrc\regim
git add . -A
git commit -m %1
git push

cd D:\GitHubSrc\tensorwatch
git add . -A
git commit -m %1
git push

popd