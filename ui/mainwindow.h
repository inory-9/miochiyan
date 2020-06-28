#pragma once

#include <QPointer>

class QMessageBox;

#include "ui_mainwindow.h"

class HMainwindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit HMainwindow(QWidget *parent = 0);
    virtual ~HMainwindow();

private:
    std::unique_ptr<Ui::MainWindow> ui;
};


