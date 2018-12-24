package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.io.resource.Resource;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;
import com.gengoai.stream.SparkStream;
import com.gengoai.stream.StreamingContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DistributedCSVDataSource implements DataSource {
   private final boolean hasHeader;
   private final int labelColumnIndex;
   private final String labelColumn;

   public DistributedCSVDataSource(int lblIndex, boolean hasHeader) {
      this.hasHeader = hasHeader;
      this.labelColumn = null;
      this.labelColumnIndex = lblIndex;
   }

   public DistributedCSVDataSource(String labelColumn, boolean hasHeader) {
      this.hasHeader = hasHeader;
      this.labelColumnIndex = -1;
      this.labelColumn = labelColumn;
   }

   public DistributedCSVDataSource(boolean hasHeader) {
      this(0, hasHeader);
   }

   @Override
   public MStream<Example> stream(Resource location) throws IOException {
      SQLContext sqlContext = new SQLContext(StreamingContext.distributed().sparkSession());
      org.apache.spark.sql.Dataset<Row> rows = sqlContext.read().option("header", hasHeader).csv(location.path());
      List<String> headers = Arrays.asList(rows.columns());
      final int li = labelColumnIndex >= 0
                     ? labelColumnIndex
                     : headers.indexOf(labelColumn);
      return new SparkStream<>(rows.toJavaRDD()
                                   .map(row -> {
                                      List<Feature> features = new ArrayList<>();
                                      String label = li >= 0 ? row.getString(li) : null;
                                      for (int i = 0; i < row.size(); i++) {
                                         if (i != li) {
                                            while (headers.size() < i) {
                                               headers.add("AutoColumn_" + i);
                                            }
                                            Double value = Math2.tryParseDouble(row.getString(i));
                                            if (value != null) {
                                               features.add(Feature.realFeature(headers.get(i), value));
                                            } else {
                                               features.add(
                                                  Feature.realFeature(headers.get(i) + "=" + row.get(i), 1.0));
                                            }
                                         }
                                      }
                                      return new Instance(label, features);
                                   }));
   }
}//END OF DistributedCSVDataSource
