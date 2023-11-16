module.exports = {
    mkdir: true, // 新增配置  是否显示目录
    // 新增配置   是否显示封面  为ture时必须在根目录设置facebook.html文件（文件名可以改，与facebookName字段一致即可）
    facebook: true,
    facebookName: "facebook.html", // 新增配置  封面的源文件名称     facebook为true时 此字段必须配置
    contents: ["D:/blog/docs/_sidebar.md"], // 需要转换文件目录，自动追踪链接文件（建议此配置为绝对路径）
    pathToPublic: "pdf/icsdoc.pdf", // 生成pdf存放的路径
    pdfOptions: {
      format: 'A4',
      displayHeaderFooter: true,
      headerTemplate: `<span>title</span>`,
      footerTemplate: `<div style='text-align:center;width: 297mm;font-size: 10px;'><span class='pageNumber'>inspur</span></div>`,
      margin: {
        top: '50px',
        right: '20px',
        bottom: '40px',
        left: '20px'
      },
      printBackground: true,//打印背景
      omitBackground: true,
      landscape: false,//纸张方向        
    }, // reference: https://github.com/GoogleChrome/puppeteer/blob/master/docs/api.md#pagepdfoptions
    removeTemp: true, // remove generated .md and .html or not
    emulateMedia: "screen", // mediaType, emulating by puppeteer for rendering pdf, 'print' by default (reference: https://github.com/GoogleChrome/puppeteer/blob/master/docs/api.md#pageemulatemediamediatype)
  }