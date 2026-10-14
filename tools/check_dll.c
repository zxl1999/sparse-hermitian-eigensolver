#include <windows.h>
#include <stdio.h>
int main(int argc, char **argv) {
  for (int i=1;i<argc;i++) {
    HMODULE h=LoadLibraryA(argv[i]);
    if(!h){DWORD e=GetLastError(); char b[1024]={0}; FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,NULL,e,0,b,sizeof(b),NULL); printf("%s FAIL error=%lu %s\n",argv[i],(unsigned long)e,b);}
    else { printf("%s OK %p\n",argv[i],(void*)h); }
  }
}
