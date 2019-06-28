
base64enc
--------
支持 Base16 /32 / 64 / 85 编解码，与系统自带的base64区别是能够识别字符集编码；

chardet
--------
尝试识别文件的字符集编码；

charsets
--------
查找字符集编码名称，支持模糊匹配，用于编码转换时的from / to；

dateparser
--------
识别各类文字描述的日期时间，转换为统一格式及时区信息；

errorcodes
--------
查找errno，支持模糊匹配；

ffenc_converter
--------
批量转换文件格式及文件编码，转换CRLF(windows)或CR(Mac)为LF(Unix)，转换编码为UTF-8；

htmlenc
--------
识别html文件里的charset属性值；

httpstatus
--------
查找HTTP状态码，支持模糊匹配；

ipcalculator
--------
IP / CIDR / Subnet 计算；

mimetypes
--------
查找文件所属mimetype，支持模糊匹配；

platforminfo
--------
列出系统配置及开发环境；

service_monitor
--------
监控服务自动拉起；

services
--------
查找 service / protocol / port，支持模糊匹配；

tattoos
--------
常见魔数；

textmatch
--------
文件内容相似性比较；

timezones
--------
查找或列出全球时区信息，支持模糊匹配；

urlparser
--------
urlencode编解码、URL格式解析和重组、Querystring解析和重组；


实现
--------
* **Python3**：base64enc、chardet、charsets、dateparser、errorcodes、htmlenc、httpstatus、ipcalculator、mimetypes、services、tattoos、textmatch、timezones、urlparser
* **Bash**：ffenc_converter、service_monitor、platforminfo
