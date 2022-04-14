<!--
 * @Date: 2022-03-16 10:15:45
 * @LastEditors: yhxiong
 * @LastEditTime: 2022-03-23 09:45:20
 * @Description: 
-->
## 数据说明
```
 visualize_tool
    ├── function.js
    ├── layout_result.css
    ├── order.js
    ├── order.json
    ├── README.md
    ├── result_refine.js
    ├── test_queries.js
    ├── test_queries.json
    ├── video_mp4
    └── visualize.html
```

* `order.js`：存放视频id和track_id映射的json 
* `test_queries.js`：存放test_queries.json
* `result_fine.js`：存放模型预测结果的json
* `video_mp4`:存放视频的文件夹
  
## 运行

1. 安装live server插件,setting.json中添加`"liveServer.settings.ignoreFiles":["**"]`
2. `result_fine.js`、`test_queries.js`、`order.js`数据替换
3. 视频文件夹`video_mp4`放于当前目录下
4. 用live server打开`visualize.html` (建议用chrome)