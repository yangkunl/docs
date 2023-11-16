(function () {
    var timer = null
    //页面初始化完成
    $(document).ready(function () {
      //监听 #main 内容发生改变
      $("#main").on('DOMNodeInserted', setHighlight)
    })
   
    /**
     * 在主内容（#main）中设置高亮
     * 这里timeout做防抖
     */
    function setHighlight() {
      timer && clearTimeout(timer)
      timer = setTimeout(function () {
        var search = getSearch().s
        if (search) {
          //去掉上次匹配关键词的样式
          $("#main .search-keyword").removeClass("search-keyword")
          var newHtml = getHtmlStr($("#main").html(), search)
          //先解绑dom更新监听  待设置高亮完成后再重新注册监听  否则会进入死循环
          newHtml && $("#main").off('DOMNodeInserted').html(newHtml).on('DOMNodeInserted', setHighlight)
        }
   
      }, 200)
    }
    /**
     * 忽略命中html标签内的属性 
     * 分别对<、>的位置进行判断 区分关键字是否出现在html标签中间，
     * 如果是     跳过 不替换
     * 如果不是   替换高亮展示
     */
    function getHtmlStr(htmlStr, search) {
      //用关键字 区分大小写 分割字符串
      var reg = new RegExp(search, 'i')
      var tempList = htmlStr.split(reg)
   
      var newHtml = tempList.shift()
      var newHtmlIndex = newHtml === undefined ? 0 : newHtml.length
   
      tempList.map(temp => {
        var strLastLeft = newHtml.lastIndexOf("<")
        var strLastRight = newHtml.lastIndexOf(">")
   
        if (strLastRight < strLastLeft) {//最后的标签未闭合
          newHtml = newHtml + search + temp
        } else {
          // 获取原字符串中的关键字  为了区分大小写显示   添加高亮样式
          newHtml = newHtml + "<em class='search-keyword'>" + htmlStr.substr(newHtmlIndex, search.length) + "</em>" + temp
        }
        //计算原关键字在原字符串中的位置
        newHtmlIndex = newHtmlIndex + search.length + temp.length
      })
      return newHtml
    }
    /**
     * 获取搜索关键字
     */
    function getSearch() {
      let hash = location.hash
      let search = hash && hash.indexOf("?") > -1 && hash.substring(hash.indexOf("?") + 1)
      let searchObj = {}
      if (search) {
        search.split("&").map(key_value => {
          let temp = key_value.split("=")
          searchObj[temp[0]] = decodeURI(temp[1])
        })
      }
      return searchObj
    }
   
  }())