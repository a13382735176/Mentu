to do with hdfs,to do with hadoop,to do with bgl...  
there are the same model，
Due to the different split ratios for the training and test sets, I have separated the code into three parts. Moreover, regarding Hadoop and Spark, the original data uses 1 to represent an anomaly and 0 to represent normal, which contradicts our convention where 1 indicates normal and 0 indicates an anomaly. Therefore, in the code, I converted it with data['Status'] = data['Status'].apply(lambda x: 1 if x == 0 else 0) to avoid confusion, and I encapsulated this in a separate Python file.

Additionally, the only thing you need to change is the threshold:
----
bgl 20l throld 95
bgl 200l throld 95
bgl100l throld 95.8
-------------
hdfs 99.3
---------------
hadoop2 97.35
-------------
hadoop3 96.3
---
spark2/spark3   100(not need bert)
----
 TB 20L 100
 TB 100L 99.99
TB 200L 100
----
Spirit 20l 98
Spirit 100l 99
Spirit 200l 98
------------------------------------------
Attention，if throld=100，
elif score>throld:                                                                                                   elif score>=throld:
        m.append(0)  # 异常  -----------》》》》》》》change to right（add a =）
        print(1)
                                                                                                                                   m.append(0)  # 异常   
                                                                                                                                         print(1)
---------------------------------------------------------------------------
Additionally, it is suggested to run Spirit last, because it was not expected to surpass the baseline to begin with.