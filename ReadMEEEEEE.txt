這是distributed real-time system的簡易說明檔

Distributed Real-Time System
	-ByteTrack
	-fast-reid
	-Dev

1. "ByteTrack"及"fast-reid"是我們系統中引用他人作品的程式碼，安裝方式詳見他們的官網，請自己GOOGLE。
   未來的應用中可以自行更換其他的演算法進來，需要將核心程式碼複製進"Dev"資料夾中並更改程式碼，其他作品的程式碼架構可能有所不同，請自行看懂新程式碼並接近我的系統。
2. "Dev"資料夾下是分散式即時跨相機多目標追蹤系統的程式碼，會在下面做介紹。


Dev
	-fastreid
	-reid_weight
	-yolox相關資料夾
	-system_output
	-setting_Gateway.json
	-camera_(detection/track/MCT).py
	-base_camera.py
	-pipeline.py
	-GMM.py
	-run_all.sh
	-gen_type.py

1."fastreid"
    這份資料夾中是fastreid的核心程式，因為在pipeline中會使用到所以複製進來。

2."reid_weight"
    這份資料夾底下是訓練好的reid模型，目前這份程式碼是基於NTU-MTMC來建立的，如果需要用到其他資料集，請自行訓練。

3."yolox"."yolox_weight"...
    這些資料夾中是yolox的相關程式碼，因為在pipeline中會使用到所以複製進來。
    裡面一樣有訓練好的detection模型，是基於NTU-MTMC訓練的，訓練過程依據yolox的github指示，如果需要應用到其他地方請自行研究。

4."system_output"
    這個資料夾是整個系統的輸出資料夾，輸出的檔案名稱會是"CamX.txt"，對應到該相機下的追蹤結果。

5."setting_Gateway.json"
    這些檔案是屬於設定檔，由於我設計的是分散式系統，所以最佳模擬狀況需要在多個裝置下進行，這裡是根據NTU-MTMC建立，所以一共需要4個gateway而有四個檔案。
    每個檔案中會有此gateway所管轄的相機名稱、相機編號。除此之外，由於此系統需要在gateway之間傳輸訊息，所以需要開通SCP的port來進行傳輸，每個port只會對應到一個傳輸對象。
    此檔案中一樣有列出對應對象的ip以及為其開通的接收port。
    如果要應用到其他資料集，需要自行設計每台裝置管理的相機及裝置間的傳輸關係，同時也要在選定的裝置(e.g. RTi2.Pro.MediaX.RTi)上設定好相關的port開通。

6."base_camera.py"
    這個檔案是所有pipeline架構的基礎定義class，如果需要增加新的pipeline stage需要在這邊新增，同時關於跨相機追蹤的演算法也寫成module包在這裡。
    每階段pipeline之間buffer所存的資料格式也是在這裡做更改。

7."camera_(detection/track/MCT).py"
    這些檔案會從"base_camera.py"中呼叫class後去建立對應的功能，實際上呼叫detection或是Reid或是MCT都是在這裡呼叫。
    init()中會先初始化model.setting.timer等物件，等到下一階段來使用。如果要使用tensorRT相關的model，在初始化上會有相關問題，可以參考我在程式碼中的解法。
    XXX_frames()中會先從前一個pipeline stage的buffer去取得其輸出資訊，在根據此階段任務去執行演算法，最後將資訊丟到buffer中。

8."pipeline.py"
    這個檔案是所有程式碼的頂端，執行時也是從這裡執行，相關的參數設定也從這裡輸入。
    這程式碼會根據json檔建立出對應的gateway及其附屬架構，由於我們程式碼中是以pipeline的架構為主，在程式碼中會以multi-thread來模擬，所以會需要等待所有streamer都建立成功才會開始處理。
    
9."GMM.py"
    根據Gaussian Mixture Model搭配上training set中跨相機移動的數據去估算出兩種類別的移動時間分布，可以再根據找出來的高斯分布去設定time window function。

10."gen_type.py"
    此檔案會用CNN來預測各bounding box所對應類別是行人還是腳踏車，模型是我自己訓練的，要用到其他地方自已研究。


11."run_all.sh"
    這是我懶得每次都打指令才寫好的bash，目前裡面的指令可以讓你在一台裝置上跑所有的資料，但請記得先設定好資料集的路徑及其他的環境架設。
    同時這個系統本身不是設計在一個裝置上跑的，所以你需要重新更改json檔，然後跑的這台電腦可能會跑超久超卡，請自行斟酌。
    我當初是跑在四台不同的裝置上且沒有其他人同時在使用這些裝置，所以速度才會出來，想模擬出一樣的狀況就求佛吧，現在實驗室的工作站越來越舊了。

12."requirements.txt"
    這份程式碼本身不會用到太多的環境，所以建立環境時主要要滿足的是yolox及bytetrack，安裝方法請依照他們網站的步驟。
    剩下會有一些環境系統喊缺什麼裝什麼就好。
	






























