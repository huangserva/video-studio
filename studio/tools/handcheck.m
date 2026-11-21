#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        for (int j=1; j<argc; j++) {
            NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[j]]];
            AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
            AVAssetImageGenerator *gen = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
            gen.appliesPreferredTrackTransform = YES;
            gen.maximumSize = CGSizeMake(640,640);
            gen.requestedTimeToleranceBefore = kCMTimeZero;
            gen.requestedTimeToleranceAfter = kCMTimeZero;
            VNDetectHumanHandPoseRequest *req = [VNDetectHumanHandPoseRequest new];
            req.maximumHandCount = 2;
            NSMutableArray *records = [NSMutableArray new];
            int count = floor(CMTimeGetSeconds(asset.duration)*8);
            for (int i=0; i<count; i++) {
                @autoreleasepool {
                    CGImageRef img = [gen copyCGImageAtTime:CMTimeMake(i,8) actualTime:NULL error:nil];
                    if (!img) continue;
                    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:img options:@{}];
                    [handler performRequests:@[req] error:nil];
                    CGImageRelease(img);
                    NSMutableArray *hands = [NSMutableArray new];
                    for (VNHumanHandPoseObservation *h in req.results) {
                        NSDictionary *pts = [h recognizedPointsForGroupKey:VNHumanHandPoseObservationJointsGroupNameAll error:nil];
                        int n=0;
                        CGFloat x=0,y=0,x0=1,y0=1,x1=0,y1=0;
                        for (VNRecognizedPoint *p in pts.allValues) {
                            if (p.confidence < .35) continue;
                            CGFloat xx=p.location.x, yy=1-p.location.y;
                            x+=xx; y+=yy; n++;
                            x0=MIN(x0,xx); x1=MAX(x1,xx); y0=MIN(y0,yy); y1=MAX(y1,yy);
                        }
                        if (n>=9) [hands addObject:@{@"x":@(x/n),@"y":@(y/n),@"points":@(n),@"span":@(MAX(x1-x0,y1-y0))}];
                    }
                    [records addObject:@{@"t":@(i/8.0),@"hands":hands}];
                }
            }
            NSData *data=[NSJSONSerialization dataWithJSONObject:records options:0 error:nil];
            printf("%s\n",[[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] UTF8String]);
        }
    }
    return 0;
}
