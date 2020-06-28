#include "mainwindow.h"
#include "core/expression_capture.h"


#include <QApplication>


#if _WIN32
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" )
#endif

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    expc::init();

    HMainwindow window;
    window.show();

    return a.exec();
}
