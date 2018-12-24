package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.io.CSV;
import com.gengoai.io.CSVReader;
import com.gengoai.io.resource.Resource;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The type Csv data source.
 *
 * @author David B. Bracewell
 */
public class CSVDataSource implements DataSource {
   private final CSV csv;
   private final String labelColumn;
   private final int labelColumnIndex;

   /**
    * Instantiates a new Csv data source.
    *
    * @param csv the csv
    */
   public CSVDataSource(CSV csv) {
      this(csv, 0);
   }

   /**
    * Instantiates a new Csv data source.
    *
    * @param csv        the csv
    * @param labelIndex the label index
    */
   public CSVDataSource(CSV csv, int labelIndex) {
      this.csv = csv;
      this.labelColumn = null;
      this.labelColumnIndex = labelIndex;
   }

   /**
    * Instantiates a new Csv data source.
    *
    * @param csv         the csv
    * @param labelColumn the label column
    */
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
                     while (headers.size() <= i) {
                        headers.add("AutoColumn_" + i);
                     }
                     Double value = Math2.tryParseDouble(row.get(i));
                     if (value != null) {
                        features.add(Feature.realFeature(headers.get(i), value));
                     } else {
                        features.add(Feature.realFeature(headers.get(i) + "=" + row.get(i), 1.0));
                     }
                  }
               }
               examples.add(new Instance(label, features));
            }
         });
      }
      return StreamingContext.local().stream(examples);
   }


}//END OF CSVDataSource
