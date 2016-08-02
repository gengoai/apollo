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

  public LibSVMDataSource(@NonNull Resource resource) {
    super(resource);
  }

  @Override
  public MStream<Instance> stream() throws IOException {
    return getStreamingContext().textFile(getResource())
      .map(line -> {
        String[] parts = line.split("\\s+");
        List<Feature> featureList = new ArrayList<>();
        Object target;
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
