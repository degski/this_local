
 @echo off

 set "HOME=%CD%"
 cd ..

 cd hedley
 git pull
 copy /B /Y hedley.h %HOME%\include\this_local\this_local\hedley.h > NUL
 cd ..

 cd plf_colony
 git pull
 copy /B /Y plf_colony.h %HOME%\include\this_local\this_local\plf_colony.h > NUL
 cd ..

 cd plf_list
 git pull
 copy /B /Y plf_list.h %HOME%\include\this_local\this_local\plf_list.h > NUL
 cd ..

 cd plf_stack
 git pull
 copy /B /Y plf_stack.h %HOME%\include\this_local\this_local\plf_stack.h > NUL
 cd ..

 cd %HOME%

 echo done
 pause
