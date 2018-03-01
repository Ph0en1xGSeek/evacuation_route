mapboxgl.accessToken = 'pk.eyJ1IjoicGgwZW4xeGdzZWVrIiwiYSI6ImNqY216bmpiejEzMTcycXBmem45YXFzZjUifQ.qEJRPYjore_nSlHfA42d0g';

$.getJSON('line_data_MDP2.json', function (datas) {
    console.log(datas)
        var szRoad = {
        success: true,
        errorCode: 0,
        errorMsg: "成功",
        data: datas
    }

    var taxiRoutes = [];
    var data = szRoad.data;
    var hStep = 300 / (data.length - 1);

    var i = 0;
    for (var x in data) {
        // i++;
        // if(i<5000)
        //     continue;
        var line = data[x];
        // if(busLines.length>500)
        //     break;
        var pointString = line.ROAD_LINE;
        var pointArr = pointString.split(';');
        var lnglats = [];
        for (var j in pointArr) {
            lnglats.push(pointArr[j].split(','))
        }
        taxiRoutes.push({
            coords: lnglats,
            lineStyle: {
                // color: echarts.color.modifyHSL('#5A94DF', Math.round(hStep * x))
            }
        })
    }

    var chart = echarts.init(document.getElementById('main'));
    chart.setOption({
        mapbox: {
            center: [141, 36.5],
            zoom: 8,
            // pitch: 50,
            // bearing: -10,
            altitudeScale: 10000000,
            style: 'mapbox://styles/mapbox/dark-v9',
            postEffect: {
                enable: true,
                FXAA: {
                    enable: true
                }
            },
            light: {
                main: {
                    intensity: 1,
                    shadow: true,
                    shadowQuality: 'high'
                },
                ambient: {
                    intensity: 0.
                }
                // ambientCubemap: {
                //     texture: '/asset/get/s/data-1491838644249-ry33I7YTe.hdr',
                //     exposure: 1,
                //     diffuseIntensity: 0.5,
                //     specularIntensity: 2
                // }
            }
        },
        series: [{
        type: 'lines3D',

        coordinateSystem: 'mapbox',

        effect: {
            show: true,
            constantSpeed: 10,
            trailWidth: 3,
            trailLength: 1,
            trailOpacity: 1,
            spotIntensity: 10
        },

        blendMode: 'lighter',

        polyline: true,

        lineStyle: {
            width: 1,
            color: 'rgb(200, 40, 0)',
            opacity: 0.
        },

        data: {
            count: function () {
                return taxiRoutes.length;
            },
            getItem: function (idx) {
                return taxiRoutes[idx]
            }
        }
    }]
    }
    )


})



