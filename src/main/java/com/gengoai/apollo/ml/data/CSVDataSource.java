package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.io.CSV;
import com.gengoai.io.CSVReader;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class CSVDataSource implements DataSource {
   private final CSV csv;
   private final String labelColumn;
   private final int labelColumnIndex;

   public CSVDataSource(CSV csv) {
      this(csv, 0);
   }

   public CSVDataSource(CSV csv, int labelIndex) {
      this.csv = csv;
      this.labelColumn = null;
      this.labelColumnIndex = labelIndex;
   }

   public CSVDataSource(CSV csv, String labelColumn) {
      this.csv = csv;
      this.labelColumn = labelColumn;
      this.labelColumnIndex = -1;
   }


   @Override
   public MStream<Example> stream(Resource location) throws IOException {
      List<Example> examples = new ArrayList<>();
      try (CSVReader reader = csv.reader(location)) {
         final List<String> headers = reader.getHeader().isEmpty()
                                      ? new ArrayList<>()
                                      : reader.getHeader();
         final int li = labelColumnIndex >= 0
                        ? labelColumnIndex
                        : headers.indexOf(labelColumn);

         reader.forEach(row -> {
            if (row.size() > 0) {
               List<Feature> features = new ArrayList<>();
               String label = li >= 0 ? row.get(li) : null;
               for (int i = 0; i < row.size(); i++) {
                  if (i != li) {
                     while (headers.size() < i) {
                        headers.add("AutoColumn_" + i);
                     }
                     features.add(Feature.realFeature(headers.get(i), Double.parseDouble(row.get(i))));
                  }
               }
               examples.add(new Instance(label, features));
            }
         });
      }
      return StreamingContext.local().stream(examples);
   }


//   private MStream<Example> distributed(Resource location) throws IOException {
//      JavaSparkContext sc = StreamingContext.distributed().sparkContext();
//      SQLContext sqlContext = new SQLContext(sc);
//      sqlContext.read().option("header", csv.getHasHeader()).csv(location.path());
//   }

   @Override
   public void write(Resource location, Dataset dataset) throws IOException {

   }
}//END OF CSVDataSource
