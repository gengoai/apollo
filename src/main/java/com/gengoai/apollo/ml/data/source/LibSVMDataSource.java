package com.gengoai.apollo.ml.data.source;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.mango.io.resource.Resource;
import com.gengoai.mango.stream.MStream;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import lombok.Getter;
import lombok.NonNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>Reads datasets in LibSVM format</p>
 *
 * @author David B. Bracewell
 */
public class LibSVMDataSource extends DataSource<Instance> {
   private static final long serialVersionUID = 1L;
   @Getter
   private final boolean multiclass;

   /**
    * Instantiates a new Lib svm data source.
    *
    * @param resource the resource
    */
   public LibSVMDataSource(@NonNull Resource resource) {
      this(resource, false);
   }

   /**
    * Instantiates a new Lib svm data source.
    *
    * @param resource   the resource
    * @param multiclass True if the dataset is multiclass, False if it is binary
    */
   public LibSVMDataSource(@NonNull Resource resource, boolean multiclass) {
      super(resource);
      this.multiclass = multiclass;
   }

   @Override
   public MStream<Instance> stream() throws IOException {
      return getStreamingContext().textFile(getResource())
                                  .map(line -> {
                                     String[] parts = line.split("\\s+");
                                     List<Feature> featureList = new ArrayList<>();
                                     Object target;
                                     if (multiclass) {
                                        target = parts[0];
                                     } else {
                                        switch (parts[0]) {
                                           case "+1":
                                           case "1":
                                              target = "true";
                                              break;
                                           case "-1":
                                              target = "false";
                                              break;
                                           default:
                                              target = parts[0];
                                              break;
                                        }
                                     }
                                     for (int j = 1; j < parts.length; j++) {
                                        String[] data = parts[j].split(":");
                                        int fnum = Integer.parseInt(data[0]) - 1;
                                        double val = Double.parseDouble(data[1]);
                                        featureList.add(Feature.real(Integer.toString(fnum), val));
                                     }
                                     return Instance.create(featureList, target);
                                  });
   }

}// END OF LibSVMDataSource
