#include "mainwindow.h"

#include "core/expression_capture.h"

#include <QFileDialog>
#include <QMainWindow>
#include <QPushButton>
#include <QTimer>
#include <QDebug>

QImage cvMat2QImage(const expc::M_Frame &frame)
{
    //qDebug() << frame.type;

    if (!frame.data)
        return QImage();

    switch (frame.type) {
    case 3: {
        QImage image(frame.width, frame.height, QImage::Format_Indexed8);
        image.setColorCount(256);

        for (int i = 0; i < 256; i++) {
            image.setColor(i, qRgb(i, i, i));
        }
        const uchar *pSrc = frame.data;
        for (int k = 0; k < frame.height; k++) {
            uchar *pDest = image.scanLine(k);
            memcpy(pDest, pSrc, uint64_t(frame.width));
            pSrc += frame.step;
        }
        return image;
    }
    case 16: {
        return QImage(frame.data, frame.width, frame.height, frame.step, QImage::Format_RGB888);
    }
    case 24: {
        return QImage(frame.data, frame.width, frame.height, frame.step, QImage::Format_ARGB32);
    }
    default:
        return QImage();

    }
}

HMainwindow::HMainwindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setCentralWidget(ui->widget);

    expc::open_camera();

    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [ = ]() {
        auto image = expc::get_image_predicted();

        auto q_image = cvMat2QImage(image.first);
        if (!q_image.isNull())
            ui->label_camera->setPixmap(QPixmap::fromImage(q_image));

        q_image = cvMat2QImage(image.second);
        if (!q_image.isNull())
            ui->label_result->setPixmap(QPixmap::fromImage(q_image));
    });

    connect(ui->pushButton_load, &QPushButton::clicked, this, [ = ]() {
        QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), nullptr, tr("Images (*.png)"));

        if (fileName.isEmpty())
            return;

        QImage image;
        if (image.load(fileName)) {
            ui->label_image->setPixmap(QPixmap::fromImage(image));
            expc::set_source_image(fileName.toStdString());
        }
    });

    connect(ui->pushButton_open, &QPushButton::clicked, this, [ = ]() {
        timer->start(100);
    });

}

HMainwindow::~HMainwindow()
{
    expc::uninit();
}
