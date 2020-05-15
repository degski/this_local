
 @echo off

 set "HOME=%CD%"
 cd ..

 cd hedley
 git pull
 copy /B /Y hedley.h %HOME%\include\this_local\this_local\hedley.h > NUL
 cd ..

 cd plf_list
 git pull
 copy /B /Y plf_list.h %HOME%\include\this_local\this_local\plf_list.h > NUL
 cd ..

 cd %HOME%

 echo done
 pause
