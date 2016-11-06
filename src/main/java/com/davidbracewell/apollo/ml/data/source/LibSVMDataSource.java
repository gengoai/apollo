package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class LibSVMDataSource extends DataSource<Instance> {
   private static final long serialVersionUID = -8917299785352538595L;
   public final boolean multiclass;

   public LibSVMDataSource(@NonNull Resource resource) {
      this(resource, false);
   }

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
